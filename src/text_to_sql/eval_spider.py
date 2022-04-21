from eval_exec.db_schema import DBSchema
from eval_exec.predicted_sql import fix_sql_casing
from eval_exec.qdmr_encoding_parser import formula_to_ref_encoding
from eval_exec.qdmr_sql import qdmr_to_sql, normalize_qdmr_prediction, prune_nested_queries
from eval_exec.sql_execution import correct_denotation
from eval_exec.utils import get_table_and_column
from evaluation import *
from process_sql import *
import re


def restore_oov(prediction):
    """
    Replace T5 SPM OOV character with `<`.
    Certain punctuation characters are mapped to the OOV symbol in T5's
    sentence-piece model. For Spider, this appears to only affect the `<` symbol,
    so it can be deterministically recovered by running this script.
    An alternative is to preprocess dataset to avoid OOV symbols for T5.
    """
    pred = prediction.replace(" ⁇ ", "<")
    return pred


def format_sql(prediction, no_split=None):
    pred = restore_oov(prediction)
    if no_split:
        return pred
    return pred.split(";")[0] + ";"


UPPERCASE_DBS = ["cre_Doc_Control_Systems",
                 "cre_Doc_Template_Mgt",
                 "cre_Doc_Tracking_DB",
                 "cre_Docs_and_Epenses",
                 "cre_Drama_Workshop_Groups",
                 "cre_Theme_park",
                 "insurance_and_eClaims"]


class ExactSetMatch(object):

    def __init__(self):
        return

    def extract_db_name(self, example):
        db = example.split(":")[0].strip()
        for uppercase_db in UPPERCASE_DBS:
            if uppercase_db.lower() == db:
                return uppercase_db
        return db

    def evaluate(self, examples, gold, predict, db_dir, table, etype):
        kmaps = build_foreign_key_map_from_json(table)
        glist = gold
        plist = predict
        db_list = [self.extract_db_name(ex) for ex in examples]
        evaluator = Evaluator()

        levels = ['easy', 'medium', 'hard', 'extra', 'all']
        partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                         'group', 'order', 'and/or', 'IUEN', 'keywords']
        entries = []
        scores = {}

        for level in levels:
            scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
            scores[level]['exec'] = 0
            for type_ in partial_types:
                scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'f1': 0., 'acc_count': 0, 'rec_count': 0}

        eval_err_num = 0
        for p, g, db in zip(plist, glist, db_list):
            p_str = format_sql(p)
            g_str = format_sql(g)  # sentencepiece tokenization to oov tokens
            db_name = db
            db = os.path.join(db_dir, db, db + ".sqlite")
            schema = Schema(get_schema(db))
            g_sql = get_sql(schema, g_str)
            hardness = evaluator.eval_hardness(g_sql)
            scores[hardness]['count'] += 1
            scores['all']['count'] += 1

            try:
                p_sql = get_sql(schema, p_str)
            except:
                # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
                p_sql = {
                    "except": None,
                    "from": {
                        "conds": [],
                        "table_units": []
                    },
                    "groupBy": [],
                    "having": [],
                    "intersect": None,
                    "limit": None,
                    "orderBy": [],
                    "select": [
                        False,
                        []
                    ],
                    "union": None,
                    "where": []
                }
                eval_err_num += 1

            # rebuild sql for value evaluation
            kmap = kmaps[db_name]
            g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
            g_sql = rebuild_sql_val(g_sql)
            g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
            p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

            if etype in ["all", "exec"]:
                exec_score = eval_exec_match(db, p_str, g_str, p_sql, g_sql)
                if exec_score:
                    scores[hardness]['exec'] += 1.0
                    scores['all']['exec'] += 1.0

            if etype in ["all", "match"]:
                exact_score = evaluator.eval_exact_match(p_sql, g_sql)
                partial_scores = evaluator.partial_scores
                scores[hardness]['exact'] += exact_score
                scores['all']['exact'] += exact_score

        for level in levels:
            if scores[level]['count'] == 0:
                continue
            if etype in ["all", "exec"]:
                scores[level]['exec'] /= scores[level]['count']
            if etype in ["all", "match"]:
                scores[level]['exact'] /= scores[level]['count']
        if etype == "exec":
            return scores['all']['exec']
        if etype == "match":
            return scores['all']['exact']
        return {"match": scores['all']['exact'], "exec": scores['all']['exec']}


class ExecutionAccuracy(object):

    def __init__(self):
        return

    def extract_db_name(self, example):
        db = example.split(":")[0].strip()
        for uppercase_db in UPPERCASE_DBS:
            if uppercase_db.lower() == db:
                return uppercase_db
        return db

    def evaluate(self, examples, gold, predict, db_dir, exec_set_match, pred_type=None, dataset=None):
        """
        :param examples: list
            List of question and db examples
        :param gold: list
            List of cased gold SQL queries
        :param predict: list
            List of lower-cased predicted queries
        :param db_dir: string
            Path to db files
        :param exec_set_match: bool
            Flag whether to use set match when comparing execution results
        :param pred_type: str
            The format of the predictions, if not sql, convert them to sql
        :return: dict
            Dictionary with the execution accuracy & number of errors
        """
        if pred_type:
            assert pred_type in ["qdmr_formula", "qdmr_steps", "qdmr_sql", "sql"]
        glist = gold
        plist = predict
        db_list = [self.extract_db_name(ex) for ex in examples]
        evaluator = Evaluator()

        levels = ['easy', 'medium', 'hard', 'extra', 'all']
        partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                         'group', 'order', 'and/or', 'IUEN', 'keywords']
        scores = {}

        for level in levels:
            scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
            scores[level]['exec'] = 0
            for type_ in partial_types:
                scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'f1': 0., 'acc_count': 0, 'rec_count': 0}

        eval_err_num = 0

        for p, g, db in zip(plist, glist, db_list):
            val = pred_type=="qdmr_steps"
            p_str = format_sql(p, no_split=(pred_type=="qdmr_steps"))
            g_str = format_sql(g)  # sentencepiece tokenization to oov tokens
            db = os.path.join(db_dir, db, db + ".sqlite")
            # TODO: commented out because of Geo880 SQL format
            # schema = Schema(get_schema(db))
            # g_sql = get_sql(schema, g_str)
            # hardness = evaluator.eval_hardness(g_sql)
            # scores[hardness]['count'] += 1
            scores['all']['count'] += 1

            # repair predicted query into executable SQL
            print("**** g_str: ", g_str)
            print("**** p_str: ", p_str)
            try:
                if pred_type in ["qdmr_formula", "qdmr_steps"]:
                    p_str = normalize_qdmr_prediction(p_str)
                    p_str = formula_to_ref_encoding(p_str) if pred_type == "qdmr_formula" else p_str
                    p_str = qdmr_to_sql(qdmr_ref_steps_encoding=p_str,
                                        question=None,
                                        db_schema_path=db,
                                        dataset=dataset)
                    p_str = prune_nested_queries(p_str, db)
                else:
                    p_str = prune_nested_queries(p_str, db, lowercased=True)
                p_sql = fix_sql_casing(p_str, g_str)

                exec_score = correct_denotation(p_sql, g_str, db, distinct=exec_set_match)
                if exec_score:
                    # scores[hardness]['exec'] += 1.0
                    scores['all']['exec'] += 1.0
            except:
                eval_err_num += 1

        for level in levels:
            if scores[level]['count'] == 0:
                continue
            scores[level]['exec'] /= scores[level]['count']
        return {"exec": scores['all']['exec'], "errors": eval_err_num}