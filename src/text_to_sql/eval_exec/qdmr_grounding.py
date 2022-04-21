from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import ngrams
from nltk import pos_tag
from nltk.metrics import *
import itertools
from collections import OrderedDict

from eval_exec.qdmr_identifier import QDMRProgramBuilder
from eval_exec.sql_parser import SQLParser
from eval_exec.utils import get_table_and_column
import wordninja  # for concatenated word splitting

import re
import spacy
# import en_core_web_lg
#
#
# nlp = en_core_web_lg.load()
STOP_WORDS = ['the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
              "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
              'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
              'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't"]


def is_float(phrase):
    try:
        float(phrase)
        return True
    except ValueError:
        return False
    return False


def is_distinct_phrase(phrase):
    tokens = phrase.split()
    filtered_toks = list(filter(lambda x: x not in STOP_WORDS, tokens))
    for tok in filtered_toks:
        if tok in ["different", "distinct", "diffrent"]:
            return True
        return False
    return False


def get_superlative(phrase):
    tokens = phrase.split()
    for tok in tokens:
        (tok, tag) = pos_tag(["the", tok])[1]
        if tag == "JJS":
            return tok
        (tok, tag) = pos_tag([tok])[0]
        if tag == "JJS":
            return tok
    return None


def is_superlative_phrase(phrase):
    return get_superlative(phrase) is not None


def is_reference(phrase):
    phrase = phrase.strip()
    return re.match("^#[0-9]*$", phrase)


def extract_refs(phrase):
    refs = []
    toks = [i.replace(",", "").strip() for i in phrase.split()]
    return list(filter(lambda x: is_reference(x), toks))


# def glove_sim(phrase, other_phrase):
#     doc_phrase = nlp(phrase)
#     doc_other = nlp(other_phrase)
#     return doc_other.similarity(doc_phrase)


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def remove_duplicate_words(phrase):
    return " ".join(unique_list(phrase.split()))


def is_select_all(phrase):
    """is a project phrase a SELECT * clause"""
    triggers = ["information", "info"]
    for token in triggers:
        if token in phrase:
            return True
    return False


def sql_query_values(sql, schema):
    """
    Returns map from sql values to their corresponding columns

    Parameters
    ----------
    sql : str
        SQL query
    schema: DBSchema
        Relevant DBSchema object

    Returns
    -------
    dict
        Map from text and number values appearing in query
        to its relevant schema column
    """
    sql_parser = SQLParser()
    return sql_parser.extract_values(sql, schema)


class QDMRGrounding(object):
    def __init__(self, qdmr, question, schema, gold_query=None):
        self.qdmr = qdmr
        if qdmr is not None:
            builder = QDMRProgramBuilder(qdmr)
            builder.build()
            self.steps = builder.steps
        self.question = question
        self.schema = schema
        # use as schema inverted index for value mapping
        self.gold_query = gold_query
        self.grounded_steps = {}
        self.steps_phrase_groundings = {}
        self.assigned_step_grounding = {}  # assigned column groundings for select, project steps
        self.ground_index = 0
        return

    def to_dict(self):
        d = {}
        d["qdmr_grounding"] = self.qdmr
        d["question"] = self.question
        d["grounded_steps"] = self.grounded_steps
        d["grounded_sql"] = self.get_grounded_sql()
        return d

    def get_grounded_sql(self):
        try:
            return self.grounded_steps[str(self.ground_index)]["SQL"]
        except:
            return None
        return None

    def assign_groundings(self, step_groundings):
        """
        Assigns specific column groundings to select
        and project steps

        Parameters
        ----------
        step_groundings : dict
            Map from step num (string) to its assigned
            column grounding

        Returns
        -------
        dict
            The column grounding assignment
        """
        self.assigned_step_grounding = step_groundings
        return self.assigned_step_grounding

    def curr_step_assigned_grounding(self):
        if self.assigned_step_grounding is None:
            return None
        step_name = str(self.ground_index + 1)
        if step_name in self.assigned_step_grounding.keys():
            assigned_element = self.assigned_step_grounding[step_name]
            return self.element_to_col(assigned_element)
        return None

    def join_tables(self, tables, join_clause, \
                    other_tables, other_join_clause, \
                    constraint=None):
        """
        Returns a unified join path between two table sets

        Parameters
        ----------
        tables : list
            List of schema tables
        join_clause : list
            List of column pairs constituting the
            join path between tables
        other_tables : list
            List of schema tables
        other_join_clause : list
            List of column pairs constituting the
            join path between other_tables
        constraint : tuple
            Pair of (t1.col1, t2.col2) of column names
            to appear in join path

        Returns
        -------
        list
            List of all tables constituting the FROM clause after
            the join. May include new tables induced by join.
        list
            List of pairs constituting the shortest join path
            between all tables
        """
        # check if one join path is subsumed by another
        join_clauses = (join_clause, other_join_clause)
        from_clauses = (tables, other_tables)
        if join_clauses != ([], []):
            # check both clauses aren't empty before checking subsumption
            for i in range(2):
                first_clause = join_clauses[i]
                first_from = from_clauses[i]
                second_clause = join_clauses[1 - i]
                second_from = from_clauses[1 - i]
                subsumed = True
                for pair in first_clause:
                    # join clause subsumption
                    if (pair not in second_clause \
                            and (pair[1], pair[0]) not in second_clause):
                        subsumed = False
                for table in first_from:
                    # table subsumption
                    if table not in second_from:
                        subsumed = False
                if subsumed:
                    return (second_from, second_clause)
        # if clauses share a table return their union:
        # [T1, T2, T3] + [T1, T4] --> [T1, T2, T3, T4]
        for t in tables:
            if t in other_tables:
                unified_join_clause = join_clause
                for pair in other_join_clause:
                    if (pair not in unified_join_clause
                            and (pair[1], pair[0]) not in unified_join_clause):
                        unified_join_clause += [pair]
                unified_tables = list(set(tables).union(set(other_tables)))
                return (unified_tables, unified_join_clause)
        # if clauses table are completely disjoint add the
        # shortest join path between two of their tables:
        # [T1, T2] + [T3, T4] --> [shortest path] + [T1, T2, T3, T4]
        shortest_join_path = None
        for t in tables:
            for other_t in other_tables:
                # optimal join path between two tables
                join_path = self.schema.optimal_join_path(t, other_t)
                if (shortest_join_path is None) or (len(join_path) < len(shortest_join_path)):
                    shortest_join_path = join_path
        # if we have a join constraint (t1.col1, t2.col2) where
        # t1 in tables & t2 in other_tables, try to find a join
        # path between t1 and t2 and see whether it's shortest
        if constraint is not None:
            t, _ = get_table_and_column(constraint[0])
            other_t, _ = get_table_and_column(constraint[1])
            assert t in tables and other_t in other_tables
            constraint_join = self.schema.optimal_join_path(t, other_t)
            if (constraint_join is not None and len(constraint_join) == len(shortest_join_path)):
                shortest_join_path = constraint_join
        assert shortest_join_path
        # add new tables, if exist
        unified_tables = list(set(shortest_join_path).union(set(tables), \
                                                            set(other_tables)))
        # add the shortest join path chain
        new_join_clause = self.schema.join_path_chain(shortest_join_path, constraint)
        prev_join_clauses = join_clause + other_join_clause
        for pair in prev_join_clauses:
            if (pair not in new_join_clause \
                    and (pair[1], pair[0]) not in new_join_clause):
                new_join_clause += [pair]
        return (unified_tables, new_join_clause)

    def value_sensitive_conditions(self, cond_list):
        value_sensitive_conds = []
        for cond in cond_list:
            if (cond[1] == "BETWEEN" or \
                    cond[2].startswith("( SELECT")):
                # between value range is without parentheses
                # or value equals subquery value
                value_sensitive_conds += [cond]
                continue
            val = cond[2].strip("'")  # remove extra ''
            new_cond = (cond[0], cond[1], "'%s'" % val) if not cond[2].strip().isdigit() else cond
            value_sensitive_conds += [new_cond]
        return value_sensitive_conds

    def adjust_like_conditions(self, cond_list):
        adjusted_conds = []
        for cond in cond_list:
            op, val = cond[1], cond[2]
            val = "%%%s%%" % val if op == "LIKE" else val
            val, op = ("%s%%" % val, 'LIKE') if op == "start" else (val, op)
            val, op = ("%%%s" % val, 'LIKE') if op == "end" else (val, op)
            adjusted_conds += [(cond[0], op, val)]
        return adjusted_conds

    def grounding_to_sql(self, op_type, select_cols=None, from_tables=None, join_pairs=None, conditions=None, \
                         aggregate=None, group=None, having=None, order_by=None, order=None, union=None, discard=None, \
                         intersect=None, final_step=None, type_phrases=None, distinct=None, projected_ref_idx=None):
        """
        Returns SQL grounded QDMR including string representing SQL query

        Parameters
        ----------
        op_type : str
            Original QDMR operation type
        select_col : list
            Columns to appear in the SELECT clause
        from_tables : list
            Tables to appear in the FROM clause
        join_pairs : list
            List of column pairs (col1, col2) constituting the
            join path between other_tables
        conditions : list
            List of triples (column, comparator, value)
            representing the WHERE clause of the query
        aggregate : str/list
            String representing a SQL aggregate operation
            empty string if no aggregate ###TODO: fix this!
        group : list
            List of columns that serve as keys for
            the GROUP BY clause of the query
        having : list
            List of triples (column, comparator, value)
            representing the HAVING clause of the query
        order_by : list
            List containing columns by serving
            as the ORDER BY statement keys
        order : list
            List containing the ORDER ASC/DESC LIMIT
            clause of the query
        union : list
            List of tuples of triples (column, comparator, value)
            representing the OR clauses of the query.
            Outer lists represent OR clauses & inner lists represent
            AND clauses: [(cond1 AND cond2) OR (cond3 AND cond4)]
        discard : list
            List containing the pair (column, subquery)
            to be discarded, "column NOT IN (subquery)"
        intersect: list
            List containing the pair (column, subquery)
            to be constrained, "column IN (subquery)"
        final_step : bool
            Flag indicating whether this is the
            final step of the QDMR
        type_phrases : tuple
            Pair containing 2 question phrases representing
            the trigger phrases, (prev. type + type, type)
            E.g., ("flights codes", "codes")
        distinct : bool
            Flag indicating whether DISTINCT should precede
            the first column in the select clause
        projected_step_idx : str
            Index of step which is the referenced step of the
            current project step being grounded (used for project self-joins)

        Returns
        -------
        dict
            SQL grounded QDMR
        """
        # Since Python functions have mutable default arguments:
        select_cols = [] if select_cols is None else select_cols
        from_tables = [] if from_tables is None else from_tables
        join_pairs = [] if join_pairs is None else join_pairs
        conditions = [] if conditions is None else conditions
        aggregate = [] if aggregate is None else aggregate
        group = [] if group is None else group
        having = [] if having is None else having
        order_by = [] if order_by is None else order_by
        order = [] if order is None else order
        union = [] if union is None else union
        discard = [] if discard is None else discard
        intersect = [] if intersect is None else intersect
        final_step = False if final_step is None else final_step
        type_phrases = ("", "") if type_phrases is None else type_phrases
        distinct = False if distinct is None else distinct
        assert op_type in ["select", "project", "filter", "aggregate", "group", \
                           "superlative", "superlative_group", "comparative", "comparative_group", \
                           "intersection", "union", "union_column", "discard", "sort", "arithmetic"]
        # adjust LIKE conditions
        like_conditions = self.adjust_like_conditions(list(dict.fromkeys(conditions)))
        like_union = [self.adjust_like_conditions(clause) for clause in union]
        # differentiate string/numeric values in conditions
        val_sensitive_conds = self.value_sensitive_conditions(like_conditions)
        val_sensitive_union = [self.value_sensitive_conditions(clause) \
                               for clause in like_union]
        sql = None
        distinct_select_cols = [col for col in select_cols]
        if distinct and select_cols != [] and op_type != "aggregate":
            distinct_select_cols[0] = "DISTINCT %s" % distinct_select_cols[0]
        if op_type == "arithmetic":
            assert len(distinct_select_cols) == 1
            sql = "SELECT %s " % distinct_select_cols[0]
        elif op_type == "select":
            assert len(from_tables) == 1
            sql = "SELECT %s " % ", ".join(distinct_select_cols)
            sql += "FROM %s " % from_tables[0]
            for cond in val_sensitive_conds:
                cond_clause = "AND" if "WHERE " in sql else "WHERE"
                sql += "%s %s %s %s " % (cond_clause, cond[0], cond[1], cond[2])
        elif op_type in ["project", "filter", "union", "union_column", \
                         "intersection", "aggregate", "superlative", "comparative", \
                         "superlative_group", "comparative_group", "sort", "discard"]:
            sql = "SELECT %s " % ", ".join(distinct_select_cols)
            if op_type == "aggregate" and distinct:
                # push DISTINCT inside SELECT AGG(DISTINCT x)
                sql = sql.replace("SELECT %s(" % aggregate.upper(), \
                                  "SELECT %s(DISTINCT " % aggregate.upper())
            sql += "FROM %s " % ", ".join(from_tables)
            if join_pairs != []:
                formatted_join_clauses = ["%s = %s" % pair for pair in join_pairs]
                sql += "WHERE %s " % " AND ".join(formatted_join_clauses)
            for cond in val_sensitive_conds:
                cond_clause = "AND" if "WHERE " in sql else "WHERE"
                sql += "%s %s %s %s " % (cond_clause, cond[0], cond[1], cond[2])
            # handle row union clause
            union_and_clauses = []
            for or_clause in val_sensitive_union:
                clause_conds = ["%s %s %s" % (cond[0], cond[1], cond[2]) for cond in or_clause]
                union_and_clauses += ["(%s)" % " AND ".join(clause_conds)]
            union_clause = " OR ".join(union_and_clauses)
            if union_clause != "":
                clause = "AND" if "WHERE " in sql else "WHERE"
                sql += "%s (%s) " % (clause, union_clause)
            for pair in discard:
                clause = "AND" if "WHERE " in sql else "WHERE"
                col, subquery = pair
                sql += " %s %s NOT IN ( %s ) " % (clause, col, subquery)
            for pair in intersect:
                clause = "AND" if "WHERE " in sql else "WHERE"
                col, subquery = pair
                sql += " %s %s IN ( %s ) " % (clause, col, subquery)
            if group != []:
                sql += " GROUP BY %s " % group[0]
            if having != []:
                sql += "HAVING %s %s %s " % having[0]
            if len(having) > 1:
                for cond in having[1:]:
                    sql += "AND %s %s %s " % cond
            if order_by != []:
                assert order != []
                sql += " ORDER BY %s " % ", ".join(order_by)
                sql += order[0]
            # TODO: handle project on group, superlative and comparative steps
        elif op_type == "group":
            sql = "SELECT %s " % ", ".join(select_cols) if not final_step \
                else "SELECT %s, %s " % (group[0], ", ".join(select_cols))  # add group key to select clause
            sql += "FROM %s " % ", ".join(from_tables)
            if len(join_pairs) > 0:
                formatted_join_clauses = ["%s = %s" % pair for pair in join_pairs]
                sql += "WHERE %s " % " AND ".join(formatted_join_clauses)
            for cond in val_sensitive_conds:
                clause = "AND" if "WHERE " in sql else "WHERE"
                sql += "%s %s %s %s " % (clause, cond[0], cond[1], cond[2])
            # handle row union clause
            union_and_clauses = []
            for or_clause in val_sensitive_union:
                clause_conds = ["%s %s %s" % (cond[0], cond[1], cond[2]) for cond in or_clause]
                union_and_clauses += ["(%s)" % " AND ".join(clause_conds)]
            union_clause = " OR ".join(union_and_clauses)
            if union_clause != "":
                clause = "AND" if "WHERE " in sql else "WHERE"
                sql += "%s (%s)" % (clause, union_clause)
            for pair in discard:
                clause = "AND" if "WHERE " in sql else "WHERE"
                col, subquery = pair
                sql += " %s %s NOT IN ( %s ) " % (clause, col, subquery)
            sql += " GROUP BY %s " % group[0]
        sql = sql.replace("  ", " ").strip()
        grounded_query = {}
        grounded_query["SQL"] = sql
        grounded_query["op"] = op_type
        grounded_query["SELECT"] = select_cols
        grounded_query["FROM"] = from_tables
        grounded_query["JOIN"] = join_pairs
        grounded_query["WHERE"] = list(dict.fromkeys(conditions))
        grounded_query["AGGREGATE"] = aggregate
        grounded_query["GROUP"] = group
        grounded_query["HAVING"] = having
        grounded_query["ORDER BY"] = order_by
        grounded_query["ORDER"] = order
        grounded_query["UNION"] = union
        grounded_query["DISCARD"] = discard
        grounded_query["INTERSECT"] = intersect
        grounded_query["type"] = type_phrases
        grounded_query["distinct"] = distinct
        grounded_query["project ref"] = projected_ref_idx
        return grounded_query

    def ground(self):
        self.grounded_steps = {}
        self.ground_index = 0
        for i, step in enumerate(self.steps):
            ground = None
            op = step.operator
            final_step = (i == len(self.steps) - 1)
            try:
                if op == "select":
                    ground = self.ground_select_step(step)
                elif op == "project":
                    ground = self.ground_project_step(step, final_step)
                elif op == "filter":
                    ground = self.ground_filter_step(step)
                elif op == "aggregate":
                    ground = self.ground_aggregate_step(step)
                elif op == "group":
                    ground = self.ground_group_step(step, final_step)
                elif op == "comparative":
                    ground = self.ground_comparative_step(step)
                elif op == "superlative":
                    ground = self.ground_superlative_step(step)
                elif op == "intersection":
                    ground = self.ground_intersection_step(step)
                elif op == "union":
                    ground = self.ground_union_step(step)
                elif op == "sort":
                    ground = self.ground_sort_step(step)
                elif op == "discard":
                    ground = self.ground_discard_step(step)
                elif op == "arithmetic":
                    ground = self.ground_arithmetic_step(step)
                else:
                    ground = None
                    assert ground is not None
            except AssertionError as error:
                return self.grounded_steps
            self.grounded_steps[str(i + 1)] = ground
            self.ground_index += 1
        return self.grounded_steps

    def set_phrase_groundings(self, phrase_groundings):
        """
        Sets the phrase groundings for the current QDMR step

        Parameters
        ----------
        phrase_groundings : list
            Ordered list of tuples (column, score) of
            phrase groundings for select/project/filter steps

        Returns
        -------
        bool
            True
        """
        curr_step = str(self.ground_index + 1)
        self.steps_phrase_groundings[curr_step] = phrase_groundings
        return True

    def ground_phrase_to_col(self, phrase, type_phrases=None, type_constraint=None):
        # ground phrase to column / table
        candidates = self.phrase_schema_candidates_NEW(phrase, \
                                                       type_constraint=type_constraint, \
                                                       type_phrases=type_phrases)
        element = candidates[0][0]
        return self.element_to_col(element)

    def element_to_col(self, element):
        # if element is a table name, ground it to default table column
        if element in self.schema.tables():
            element = self.schema.default_column(element)
        return element

    def subquery_condition(self, grounded_step, cond_col=None):
        paren_sql = "( %s )" % grounded_step["SQL"]
        col = grounded_step["SELECT"][0] if (cond_col is None) else cond_col
        return (col, "IN", paren_sql)

    def conflicting_conditions_tables(self, conditions, other_conditions):
        """
        Returns SQL grounded QDMR including string representing SQL query

        Parameters
        ----------
        conditions : list
            List of QDMR step conditions, (col, comparator, val)
        other_conditions : list
            List of another QDMR step conditions

        Returns
        -------
        list
            List of tables with contradicting conditions between the groups.
            Restricted only to contradicting comparison conditions.
            E.g.:
                (faculty.name, =, 'Howard') (faculty.name, =, 'Lillian')
                (people.age > 23) (people.age < 20)
        """
        tables = []
        for (col, comp, val) in conditions:
            for (col2, comp2, val2) in other_conditions:
                if col == col2 and ((comp == comp2 == '=' and val != val2) or \
                                    (comp == comp2 == 'IN' and val != val2) or \
                                    (is_float(val) and is_float(val2) and \
                                     ((comp in ['<', '<='] and comp2 in ['>', '>='] and float(val) <= float(val2)) or \
                                      (comp in ['>', '>='] and comp2 in ['<', '>='] and float(val) >= float(val2))))):
                    tables += [get_table_and_column(col2)[0]]
        return list(dict.fromkeys(tables))

    def has_conflicting_conditions(self, conds, other_conds):
        conflicts = self.conflicting_conditions_tables(conds, other_conds)
        return len(conflicts) > 0

    def ground_select_step(self, qdmr_step):
        phrase = qdmr_step.arguments[0]
        # handle DISTINCT in select step
        distinct = ("different" in phrase or \
                    "diffrent" in phrase or \
                    "distinct" in phrase or \
                    "distinct %s" % phrase[:-1] in self.question or \
                    "diffrent %s" % phrase[:-1] in self.question or \
                    "different %s" % phrase[:-1] in self.question)
        conditions, tables, columns = [], [], []
        val_col_list = self.ground_value_in_column(phrase)
        type_phrase = phrase
        if val_col_list != []:
            # ground select to schema entity, e.g., "return Jagadish"
            for pair in val_col_list:
                value, column = pair
                grounded_table, grounded_col = get_table_and_column(column)
                conditions += [(column, "=", value)]
                tables += [grounded_table]
                columns += ["%s.%s" % (grounded_table, grounded_col)]
                phrase_without_entity = phrase.replace(value, "").strip()
                type_phrase = grounded_table.replace("_", " ")
        else:
            # ground phrase to column / table
            element = self.curr_step_assigned_grounding()
            if element is None:
                # no assigned grounding
                element = self.ground_phrase_to_col(phrase)
                phrase_groundings = self.phrase_schema_candidates_NEW(phrase, \
                                                                      type_constraint=None, \
                                                                      type_phrases=None)
                self.set_phrase_groundings(phrase_groundings)
            grounded_table, grounded_col = get_table_and_column(element)
            tables = [grounded_table]
            columns = ["%s.%s" % (grounded_table, grounded_col)]
        # we assume a single column is returned by select steps
        return self.grounding_to_sql("select", \
                                     select_cols=list(dict.fromkeys(columns))[:1], \
                                     from_tables=list(dict.fromkeys(tables)), \
                                     conditions=conditions, \
                                     type_phrases=(type_phrase, type_phrase), \
                                     distinct=distinct)

    def ground_project_step(self, qdmr_step, final_step):
        # get phrase
        projection, ref = qdmr_step.arguments
        ref_num = ref.replace("#", "")
        ref_step = self.grounded_steps[ref_num]
        ref_column = ref_step["SELECT"][0]  # TODO: handle more than one column!
        _, ref_col_table = get_table_and_column(ref_column)
        # remove project step suffixes if exsits
        phrase = projection.replace("of #REF", "").replace("in #REF", "").replace("on #REF", "").replace("#REF",
                                                                                                         "").strip()
        phrase = phrase[3:].strip() if phrase.startswith("the ") else phrase
        type_phrases = ("%s %s" % (ref_step["type"][1], phrase), phrase)
        if phrase in ["different", "distinct"]:
            # if project is of type "different of #REF"
            return self.grounding_to_sql("project", \
                                         select_cols=ref_step["SELECT"], \
                                         from_tables=ref_step["FROM"], \
                                         join_pairs=ref_step["JOIN"], \
                                         conditions=ref_step["WHERE"], \
                                         group=ref_step["GROUP"], \
                                         having=ref_step["HAVING"], \
                                         order_by=ref_step["ORDER BY"], \
                                         order=ref_step["ORDER"], \
                                         union=ref_step["UNION"], \
                                         discard=ref_step["DISCARD"], \
                                         intersect=ref_step["INTERSECT"], \
                                         type_phrases=type_phrases, \
                                         distinct=True, \
                                         projected_ref_idx=ref_num)
        if final_step and is_select_all(phrase):
            # project is SELECT *
            return self.grounding_to_sql("project", \
                                         select_cols=["*"], \
                                         from_tables=ref_step["FROM"], \
                                         join_pairs=ref_step["JOIN"], \
                                         conditions=ref_step["WHERE"], \
                                         group=ref_step["GROUP"], \
                                         having=ref_step["HAVING"], \
                                         order_by=ref_step["ORDER BY"], \
                                         order=ref_step["ORDER"], \
                                         union=ref_step["UNION"], \
                                         discard=ref_step["DISCARD"], \
                                         intersect=ref_step["INTERSECT"], \
                                         type_phrases=type_phrases, \
                                         projected_ref_idx=ref_num)
        # add DISTINCT to column
        distinct = ("different" in phrase or \
                    "diffrent" in phrase or \
                    "distinct" in phrase or \
                    ref_step["distinct"] or \
                    "distinct %s" % phrase[:-1] in self.question or \
                    "diffrent %s" % phrase[:-1] in self.question or \
                    "different %s" % phrase[:-1] in self.question)
        # get potential grounding
        project_col = self.curr_step_assigned_grounding()
        conds = [self.subquery_condition(ref_step)]
        # handle project self-join, if exists
        project_self_join = self.handle_project_self_join(ref_step, type_phrases[1])
        if project_self_join is not None:
            project_col, condition = project_self_join
            conds = [condition]
        if project_col is None:
            # no assigned grounding
            project_col = self.ground_phrase_to_col(phrase, type_phrases=type_phrases)
            phrase_groundings = self.phrase_schema_candidates_NEW(phrase, \
                                                                  type_constraint=None, \
                                                                  type_phrases=type_phrases)
            self.set_phrase_groundings(phrase_groundings)
        # join the referenced query
        project_table, _ = get_table_and_column(project_col)
        constraint = (ref_column, project_col)
        from_tables, join_clause = self.join_tables(ref_step["FROM"], ref_step["JOIN"], \
                                                    [project_table], [], \
                                                    constraint=constraint)
        return self.grounding_to_sql("project", \
                                     select_cols=[project_col], \
                                     from_tables=from_tables, \
                                     join_pairs=join_clause, \
                                     conditions=conds, \
                                     type_phrases=type_phrases, \
                                     distinct=distinct, \
                                     projected_ref_idx=ref_num)

    def handle_project_self_join(self, grounded_ref_step, project_phrase):
        # Find whether it is a project self-join
        # i.e., whether the ref step type is identical to the step type.
        # E.g., "Jane Doe; father of #1; father of #2; father of #3"
        # There are two columns:
        # 1. Project column - constrained across all projections
        # 2. Condition column - the subquery condition column
        if grounded_ref_step["type"][1] == project_phrase:
            self_join_col = self.project_self_join_col(grounded_ref_step, project_phrase)
            project_col = grounded_ref_step["SELECT"][0]
            return (project_col, \
                    self.subquery_condition(grounded_ref_step, cond_col=self_join_col))
        return None

    def project_self_join_col(self, grounded_step, project_phrase):
        # Find the join condition column which is the column of the first
        # step of a different type
        if (grounded_step["type"][1] == project_phrase and grounded_step["op"] == "project"):
            ref_num = grounded_step["project ref"]
            grounded_ref_step = self.grounded_steps[ref_num]
            return self.project_self_join_col(grounded_ref_step, project_phrase)
        return grounded_step["SELECT"][0]

    def ground_project_step_OLD(self, qdmr_step, final_step):
        # get phrase
        projection, ref = qdmr_step.arguments
        ref_num = ref.replace("#", "")
        ref_step = self.grounded_steps[ref_num]
        ref_column = ref_step["SELECT"][0]  ####TODO: handle more than one column!
        _, ref_col_table = get_table_and_column(ref_column)
        # remove project step suffixes if exsits
        phrase = projection.replace("of #REF", "").replace("in #REF", "").replace("on #REF", "").replace("#REF",
                                                                                                         "").strip()
        phrase = phrase[3:].strip() if phrase.startswith("the ") else phrase
        type_phrases = ("%s %s" % (ref_step["type"][1], phrase), phrase)
        if phrase in ["different", "distinct"]:
            # if project is of type "different of #REF"
            return self.grounding_to_sql("project", \
                                         select_cols=ref_step["SELECT"], \
                                         from_tables=ref_step["FROM"], \
                                         join_pairs=ref_step["JOIN"], \
                                         conditions=ref_step["WHERE"], \
                                         group=ref_step["GROUP"], \
                                         having=ref_step["HAVING"], \
                                         order_by=ref_step["ORDER BY"], \
                                         order=ref_step["ORDER"], \
                                         union=ref_step["UNION"], \
                                         discard=ref_step["DISCARD"], \
                                         intersect=ref_step["INTERSECT"], \
                                         type_phrases=type_phrases, \
                                         distinct=True)
        if final_step and is_select_all(phrase):
            # project is SELECT *
            return self.grounding_to_sql("project", \
                                         select_cols=["*"], \
                                         from_tables=ref_step["FROM"], \
                                         join_pairs=ref_step["JOIN"], \
                                         conditions=ref_step["WHERE"], \
                                         group=ref_step["GROUP"], \
                                         having=ref_step["HAVING"], \
                                         order_by=ref_step["ORDER BY"], \
                                         order=ref_step["ORDER"], \
                                         union=ref_step["UNION"], \
                                         discard=ref_step["DISCARD"], \
                                         intersect=ref_step["INTERSECT"], \
                                         type_phrases=type_phrases)
        # add DISTINCT to column
        distinct = ("different" in phrase or \
                    "diffrent" in phrase or \
                    "distinct" in phrase or \
                    ref_step["distinct"] or \
                    "distinct %s" % phrase[:-1] in self.question or \
                    "diffrent %s" % phrase[:-1] in self.question or \
                    "different %s" % phrase[:-1] in self.question)
        # get potential grounding
        project_col = self.curr_step_assigned_grounding()
        if project_col is None:
            # no assigned grounding
            project_col = self.ground_phrase_to_col(phrase, type_phrases=type_phrases)
            phrase_groundings = self.phrase_schema_candidates_NEW(phrase, \
                                                                  type_constraint=None, \
                                                                  type_phrases=type_phrases)
            self.set_phrase_groundings(phrase_groundings)
        # join the referenced query
        project_table, _ = get_table_and_column(project_col)
        constraint = (ref_column, project_col)
        from_tables, join_clause = self.join_tables(ref_step["FROM"], ref_step["JOIN"], \
                                                    [project_table], [], \
                                                    constraint=constraint)
        if self.has_superlative_clause(ref_step):
            return self.grounding_to_sql("project", \
                                         select_cols=[project_col], \
                                         from_tables=from_tables, \
                                         join_pairs=join_clause, \
                                         conditions=[self.subquery_condition(ref_step)], \
                                         type_phrases=type_phrases, \
                                         distinct=distinct)
        return self.grounding_to_sql("project", \
                                     select_cols=[project_col], \
                                     from_tables=from_tables, \
                                     join_pairs=join_clause, \
                                     conditions=ref_step["WHERE"], \
                                     group=ref_step["GROUP"], \
                                     having=ref_step["HAVING"], \
                                     order_by=ref_step["ORDER BY"], \
                                     order=ref_step["ORDER"], \
                                     union=ref_step["UNION"], \
                                     discard=ref_step["DISCARD"], \
                                     intersect=ref_step["INTERSECT"], \
                                     type_phrases=type_phrases, \
                                     distinct=distinct)

    def ground_filter_step(self, qdmr_step):
        values, conditions = [], []
        ref, condition_phrase = qdmr_step.arguments
        ref_num = ref.replace("#", "")
        ref_step = self.grounded_steps[ref_num]
        # check if filter is distinct
        if is_distinct_phrase(condition_phrase):
            return self.grounding_to_sql("filter", \
                                         select_cols=ref_step["SELECT"], \
                                         from_tables=ref_step["FROM"], \
                                         join_pairs=ref_step["JOIN"], \
                                         conditions=ref_step["WHERE"], \
                                         group=ref_step["GROUP"], \
                                         having=ref_step["HAVING"], \
                                         order_by=ref_step["ORDER BY"], \
                                         order=ref_step["ORDER"], \
                                         union=ref_step["UNION"], \
                                         discard=ref_step["DISCARD"], \
                                         intersect=ref_step["INTERSECT"], \
                                         type_phrases=ref_step["type"], \
                                         distinct=True)
        comparator, extracted_val = extract_comparator(condition_phrase)
        assert self.gold_query is not None
        # condition has schema value string / number
        val_col_list = self.ground_value_in_column(condition_phrase)
        for pair in val_col_list:
            value, column = pair
            # make sure string values are not compared
            comparator = "=" if ((not value.isdigit()) and \
                                 comparator in [">", "<", ">=", "<="]) else comparator
            condition = (column, comparator, '%s' % value) if comparator != "BETWEEN" \
                else (column, comparator, extracted_val)
            conditions += [condition]
        if val_col_list == []:
            # extract column of a numeric value
            tokens = condition_phrase.split()
            for tok in tokens:
                if tok.isnumeric():
                    column = self.ground_phrase_to_col(condition_phrase, \
                                                       type_constraint="number", \
                                                       type_phrases=ref_step["type"])
                    # extract the condition comparator
                    comparator, value = extract_comparator(condition_phrase)
                    if value:
                        # numeric value extracted
                        conditions += [(column, comparator, value)]
        columns = [cond[0] for cond in conditions]
        # extract the type phrases
        type_phrases = ("%s %s" % (ref_step["type"][1], condition_phrase), \
                        ref_step["type"][1])
        # extract column when the condition has no value
        if conditions == []:
            # get potential grounding
            element = self.curr_step_assigned_grounding()
            if element is None:
                # ground phrase to column / table
                element = self.ground_phrase_to_col(condition_phrase, \
                                                    type_phrases=type_phrases)
                phrase_groundings = self.phrase_schema_candidates_NEW(condition_phrase, \
                                                                      type_constraint=None, \
                                                                      type_phrases=type_phrases)
                self.set_phrase_groundings(phrase_groundings)
            grounded_table, grounded_col = get_table_and_column(element)
            columns = ["%s.%s" % (grounded_table, grounded_col)]
        # join all extracted columns with the referenced query
        filter_tables = [col.split(".")[0] for col in columns]
        new_from, new_join = ref_step["FROM"], ref_step["JOIN"]
        for i in range(len(filter_tables)):
            table = filter_tables[i]
            col = columns[i]
            constraint = (ref_step["SELECT"][0], col)
            new_from, new_join = self.join_tables(new_from, new_join, \
                                                  [table], [], \
                                                  constraint=constraint)
        conds = ref_step["WHERE"] + conditions
        if self.has_conflicting_conditions(conditions, ref_step["WHERE"]):
            # self-join conflicting conditions, create subquery
            conds = conditions + [self.subquery_condition(ref_step)]
        return self.grounding_to_sql("filter", \
                                     select_cols=ref_step["SELECT"], \
                                     from_tables=new_from, \
                                     join_pairs=new_join, \
                                     conditions=conds, \
                                     group=ref_step["GROUP"], \
                                     having=ref_step["HAVING"], \
                                     order_by=ref_step["ORDER BY"], \
                                     order=ref_step["ORDER"], \
                                     union=ref_step["UNION"], \
                                     discard=ref_step["DISCARD"], \
                                     intersect=ref_step["INTERSECT"], \
                                     type_phrases=type_phrases)

    def ground_aggregate_step(self, qdmr_step):
        aggregate, ref = qdmr_step.arguments
        ref = ref.replace("#", "")
        ref_step = self.grounded_steps[ref]
        select_col = None
        select_col = "%s(%s)" % (aggregate.upper(), ref_step["SELECT"][0])
        order_by = ref_step["ORDER BY"]
        order = ref_step["ORDER"]
        if ref_step["GROUP"] != [] and ref_step["GROUP"] is not None:
            # handle aggregate on a "GROUP" subquery
            if (aggregate in ["max", "min"] or "top" in aggregate):
                select_col = ref_step["SELECT"][0]
                order_by = ref_step["SELECT"]
                order = ["DESC LIMIT 1"] if aggregate == "max" else ["ASC LIMIT 1"]
                if "top" in aggregate:
                    # extract k from top-k superlative
                    k = re.findall(r'[0-9]+', aggregate)[0]
                    order = ["DESC LIMIT %s" % k]
            else:
                return self.grounding_to_sql("aggregate", \
                                             select_cols=[select_col], \
                                             from_tables=ref_step["FROM"], \
                                             join_pairs=ref_step["JOIN"], \
                                             conditions=[self.subquery_condition(ref_step)], \
                                             aggregate=aggregate, \
                                             distinct=ref_step["distinct"])
        return self.grounding_to_sql("aggregate", \
                                     select_cols=[select_col], \
                                     from_tables=ref_step["FROM"], \
                                     join_pairs=ref_step["JOIN"], \
                                     conditions=ref_step["WHERE"], \
                                     aggregate=aggregate, \
                                     group=ref_step["GROUP"], \
                                     having=ref_step["HAVING"], \
                                     order_by=order_by, \
                                     order=order, \
                                     union=ref_step["UNION"], \
                                     discard=ref_step["DISCARD"], \
                                     intersect=ref_step["INTERSECT"], \
                                     distinct=ref_step["distinct"])

    def grounded_phrase_step(self, phrase):
        builder = QDMRProgramBuilder("return %s" % phrase)
        builder.build()
        return self.ground_select_step(builder.steps[0])

    def ground_group_step(self, qdmr_step, final_step):
        aggregate, values, keys = qdmr_step.arguments
        keys_step = self.grounded_steps[keys.replace("#", "")] if is_reference(keys) else \
            self.grounded_phrase_step(keys)  # keys is a column string: "number of #1 for each year"
        values_step = self.grounded_steps[values.replace("#", "")] if is_reference(values) else \
            self.grounded_phrase_step(values)  # values is a column string: "maximum height for each #3"
        from_tables, join_clause = self.join_tables(values_step["FROM"], values_step["JOIN"], \
                                                    keys_step["FROM"], keys_step["JOIN"])
        # TODO: handle case with multiple select columns
        return self.grounding_to_sql("group", \
                                     select_cols=["%s(%s)" % (aggregate.upper(), values_step["SELECT"][0])], \
                                     from_tables=from_tables, \
                                     join_pairs=join_clause, \
                                     conditions=values_step["WHERE"] + keys_step["WHERE"], \
                                     group=keys_step["SELECT"], \
                                     aggregate=aggregate, \
                                     final_step=final_step, \
                                     discard=values_step["DISCARD"], \
                                     intersect=values_step["INTERSECT"])

    def ground_superlative_step(self, qdmr_step, grounded_qdmr_steps=None):
        self.grounded_steps = grounded_qdmr_steps if grounded_qdmr_steps is not None else self.grounded_steps
        aggregate, keys, values = qdmr_step.arguments[:3]
        keys = keys.replace("#", "")
        values = values.replace("#", "")
        assert (aggregate in ["max", "min"] or "top" in aggregate)
        values_step = self.grounded_steps[values]
        if values_step["op"] == "group":
            return self.ground_superlative_group_step(qdmr_step)
        keys_step = self.grounded_steps[keys]
        value_col = values_step["SELECT"][0]
        # values column grounding must be a numeric column
        constraint = (value_col, keys_step["SELECT"][0])
        from_tables, join_clause = self.join_tables(values_step["FROM"], values_step["JOIN"], \
                                                    keys_step["FROM"], keys_step["JOIN"], \
                                                    constraint=constraint)
        arg_k = qdmr_step.arguments[-1] if (grounded_qdmr_steps is not None and \
                                            len(qdmr_step.arguments) == 4) else "1"
        order = [f"DESC LIMIT {arg_k}"] if aggregate == "max" else [f"ASC LIMIT {arg_k}"]
        if "top" in aggregate:
            # extract k from top-k superlative
            k = re.findall(r'[0-9]+', aggregate)[0]
            order = ["DESC LIMIT %s" % k]
        conditions = values_step["WHERE"] + keys_step["WHERE"]
        if self.has_superlative_clause(keys_step):
            # check if the keys subquery is also a superlative
            conditions = values_step["WHERE"] + [self.subquery_condition(keys_step)]
        return self.grounding_to_sql("superlative", \
                                     select_cols=keys_step["SELECT"], \
                                     from_tables=from_tables, \
                                     join_pairs=join_clause, \
                                     conditions=conditions, \
                                     order_by=values_step["SELECT"], \
                                     order=order, \
                                     type_phrases=keys_step["type"])

    def has_superlative_clause(self, step):
        if step["ORDER"] is not None:
            return (len(step["ORDER"]) > 0 and \
                    "LIMIT" in step["ORDER"][0])
        return False

    def ground_superlative_group_step(self, qdmr_step, grounded_qdmr_steps=None):
        self.grounded_steps = grounded_qdmr_steps if grounded_qdmr_steps is not None else self.grounded_steps
        aggregate, keys, values = qdmr_step.arguments[:3]
        keys = keys.replace("#", "")
        values = values.replace("#", "")
        assert (aggregate in ["max", "min"] or "top" in aggregate)
        group_step = self.grounded_steps[values]  # values refers to group step
        keys_step = self.grounded_steps[keys]
        # assert group step key is the same as the comparative key
        assert (group_step["GROUP"] == keys_step["SELECT"])
        arg_k = qdmr_step.arguments[-1] if grounded_qdmr_steps is not None else "1"
        order = [f"DESC LIMIT {arg_k}"] if aggregate == "max" else [f"ASC LIMIT {arg_k}"]
        return self.grounding_to_sql("superlative_group", \
                                     select_cols=keys_step["SELECT"], \
                                     from_tables=group_step["FROM"], \
                                     join_pairs=group_step["JOIN"], \
                                     conditions=group_step["WHERE"] + keys_step["WHERE"], \
                                     group=keys_step["SELECT"], \
                                     order_by=group_step["SELECT"], \
                                     order=order, \
                                     type_phrases=keys_step["type"])

    def ground_comparative_step(self, qdmr_step):
        keys, values, condition = qdmr_step.arguments
        keys = keys.replace("#", "")
        values = values.replace("#", "")
        comparator, value = extract_comparator(condition)
        values_step = self.grounded_steps[values]
        if values_step["op"] == "group":
            return self.ground_comparative_group_step(qdmr_step)
        keys_step = self.grounded_steps[keys]
        comparative_conds = []
        val_col_list = None
        if is_reference(value):
            # compared value is a reference to subquery
            ref = value.replace("#", "")
            value = "( %s )" % self.grounded_steps[ref]["SQL"]
        else:
            # compared value is a string - use inverted index to extract column
            # special case when value is a between statement "between 5 AND 8"
            val_col_list = self.ground_value_in_column(value) if comparator != "BETWEEN" else None
        if val_col_list:
            comparative_conds = [(column, comparator, '%s' % v) for (v, column) in val_col_list]
            cond_tables = [get_table_and_column(cond[0])[0] for cond in comparative_conds]
            from_tables, join_clause = self.join_tables(cond_tables, [], \
                                                        keys_step["FROM"], keys_step["JOIN"])
        else:
            # no columns extracted from value
            from_tables, join_clause = self.join_tables(values_step["FROM"], values_step["JOIN"], \
                                                        keys_step["FROM"], keys_step["JOIN"])
            comparative_conds = [(values_step["SELECT"][0], comparator, value)]
        conds = values_step["WHERE"] + keys_step["WHERE"] + comparative_conds
        if self.has_conflicting_conditions(comparative_conds, \
                                           values_step["WHERE"] + keys_step["WHERE"]):
            # self-join conflicting conditions, create subquery
            subquery_step = self.grounding_to_sql("comparative", \
                                                  select_cols=keys_step["SELECT"], \
                                                  from_tables=from_tables, \
                                                  join_pairs=join_clause, \
                                                  conditions=keys_step["WHERE"] + keys_step["WHERE"], \
                                                  type_phrases=keys_step["type"])
            conds = comparative_conds + [self.subquery_condition(subquery_step)]
        return self.grounding_to_sql("comparative", \
                                     select_cols=keys_step["SELECT"], \
                                     from_tables=from_tables, \
                                     join_pairs=join_clause, \
                                     conditions=conds, \
                                     type_phrases=keys_step["type"])

    def ground_comparative_group_step(self, qdmr_step):
        keys, values, condition = qdmr_step.arguments
        keys = keys.replace("#", "")
        values = values.replace("#", "")
        comparator, value = extract_comparator(condition)
        group_step = self.grounded_steps[values]  # values refers to group step
        keys_step = self.grounded_steps[keys]
        # assert group step key is the same as the comparative key
        assert (group_step["GROUP"] == keys_step["SELECT"])
        return self.grounding_to_sql("comparative_group", \
                                     select_cols=keys_step["SELECT"], \
                                     from_tables=group_step["FROM"], \
                                     join_pairs=group_step["JOIN"], \
                                     conditions=group_step["WHERE"] + keys_step["WHERE"], \
                                     group=keys_step["SELECT"], \
                                     having=[(group_step["SELECT"][0], comparator, value)], \
                                     type_phrases=keys_step["type"])

    def ground_intersection_step(self, qdmr_step, grounded_project_col=None, grounded_qdmr_steps=None):
        projection = qdmr_step.arguments[0]
        project_columns, project_table = None, None
        projection_types = (projection, projection)
        proj_refs = extract_refs(projection)
        grounded_steps = grounded_qdmr_steps if (grounded_qdmr_steps is not None) else self.grounded_steps
        if len(proj_refs) > 0:
            # check if the intersection column is a ref
            proj_ref = proj_refs[0].replace("#", "")
            step = grounded_steps[proj_ref]
            project_columns = step["SELECT"]
            project_tables = step["FROM"]
            projection_types = step["type"]
        else:
            # otherwise ground the intersection column
            project_col = grounded_project_col if (grounded_project_col is not None) \
                else self.ground_phrase_to_col(projection)
            project_table, _ = get_table_and_column(project_col)
            project_columns = [project_col]
            project_tables = [project_table]
        # join the referenced subqueries
        refs = qdmr_step.arguments[1:]
        refs = [ref.replace("#", "") for ref in refs]
        ref_steps = [grounded_steps[r] for r in refs]
        assert (len(ref_steps) == 2)
        from_tables, join_clause = self.join_tables(ref_steps[0]["FROM"], ref_steps[0]["JOIN"], \
                                                    ref_steps[1]["FROM"], ref_steps[1]["JOIN"])
        # join with the project column
        from_tables, join_clause = self.join_tables(from_tables, join_clause, \
                                                    project_tables, [])
        assert (ref_steps[0]["GROUP"] == ref_steps[1]["GROUP"] or \
                ref_steps[0]["GROUP"] == [] or \
                ref_steps[1]["GROUP"] == [])
        intersection_conditions = ref_steps[0]["WHERE"] + ref_steps[1]["WHERE"]
        # handle self join when the subqueries conditions columns should intersect
        self_join_tables = self.conflicting_conditions_tables(ref_steps[0]["WHERE"], \
                                                              ref_steps[1]["WHERE"])
        intersect = []
        if self_join_tables != []:
            subquery = self.grounding_to_sql("project", \
                                             select_cols=project_columns, \
                                             from_tables=from_tables, \
                                             join_pairs=join_clause, \
                                             conditions=ref_steps[1]["WHERE"], \
                                             group=ref_steps[1]["GROUP"], \
                                             having=ref_steps[0]["HAVING"] + ref_steps[1]["HAVING"], \
                                             intersect=ref_steps[1]["INTERSECT"], \
                                             type_phrases=projection_types)
            intersect = [(project_columns[0], subquery["SQL"])]
            intersection_conditions = ref_steps[0]["WHERE"]

        group = list(dict.fromkeys(ref_steps[0]["GROUP"] + ref_steps[1]["GROUP"]))
        return self.grounding_to_sql("intersection", \
                                     select_cols=project_columns, \
                                     from_tables=from_tables, \
                                     join_pairs=join_clause, \
                                     conditions=intersection_conditions, \
                                     group=group, \
                                     having=ref_steps[0]["HAVING"] + ref_steps[1]["HAVING"], \
                                     intersect=intersect, \
                                     type_phrases=projection_types, \
                                     distinct=True)

    def ground_union_step(self, qdmr_step, grounded_qdmr_ref_steps=None):
        ref_steps = grounded_qdmr_ref_steps
        if grounded_qdmr_ref_steps is None:
            refs = qdmr_step.arguments
            ref_steps = [self.grounded_steps[r.replace("#", "")] for r in refs]
            assert len(ref_steps) > 1
            ref_operators_set = set([step["op"] for step in ref_steps])
            if not ref_operators_set.issubset(set(["filter", "comparative"])):
                # If all subqueries not FILTER or COMPARATIVE then it is a column union,
                # e.g. return height of #1; return weight of #1; return #2, #3
                return self.ground_column_union_step(qdmr_step)
        # Otherwise it is a row union,
        # e.g. return #1 after 1999; return #1 starring Eminem; return #2, #3
        ref_selects = [step["SELECT"] for step in ref_steps]
        from_tables, join_clause = ref_steps[0]["FROM"], ref_steps[0]["JOIN"]
        for i in range(len(ref_steps) - 1):
            from_tables, join_clause = self.join_tables(from_tables, join_clause, \
                                                        ref_steps[i + 1]["FROM"], ref_steps[i + 1]["JOIN"])
        order_by, order, having, group, union = ([], [], [], [], [])
        for i in range(len(ref_steps)):
            group += ref_steps[i]["GROUP"]
            having += ref_steps[i]["HAVING"]
            order_by += ref_steps[i]["ORDER BY"]
            order += ref_steps[i]["ORDER"]
            union += ref_steps[i]["UNION"]
        # find all the common conditions across steps
        ref_conditions = [step["WHERE"] for step in ref_steps]
        joint_conds = list(set(ref_conditions[0]).intersection(*ref_conditions[1:]))
        # OR conditions are those that are unique for each step
        all_conds = list(set(ref_conditions[0]).union(*ref_conditions[1:]))
        or_conds = list(filter(lambda x: x not in joint_conds, all_conds))
        or_and_clauses = None
        if len(or_conds) > 0:
            # The two AND clauses that constitute the OR
            or_and_clauses = [tuple(set(ref_conditions[0]).intersection(set(or_conds))), \
                              tuple(set(or_conds).intersection(*ref_conditions[1:]))]
            or_and_clauses += union
            or_and_clauses = list(dict.fromkeys(or_and_clauses))
            # filter out empty clauses when intersection is empty
            or_and_clauses = list(filter(lambda x: x != (), or_and_clauses))
        # compute columns
        cols = [col for select in ref_selects for col in select]
        assert cols[0] == cols[1]
        return self.grounding_to_sql("union", \
                                     select_cols=list(set(cols)), \
                                     from_tables=from_tables, \
                                     join_pairs=join_clause, \
                                     conditions=joint_conds, \
                                     group=list(dict.fromkeys(group)), \
                                     having=list(dict.fromkeys(having)), \
                                     order_by=list(dict.fromkeys(order_by)), \
                                     order=list(dict.fromkeys(order)), \
                                     union=or_and_clauses, \
                                     type_phrases=ref_steps[0]["type"])

    def ground_column_union_step(self, qdmr_step, grounded_qdmr_ref_steps=None):
        # e.g. return height of #1; return weight of #1; return #2, #3
        ref_steps = grounded_qdmr_ref_steps
        if grounded_qdmr_ref_steps is None:
            refs = qdmr_step.arguments
            ref_steps = [self.grounded_steps[r.replace("#", "")] for r in refs]
        assert len(ref_steps) > 1
        ref_selects = [(step["GROUP"] + step["SELECT"] if step["op"] == "group" else step["SELECT"]) \
                       for step in ref_steps]  # add group key to select clause
        from_tables, join_clause = ref_steps[0]["FROM"], ref_steps[0]["JOIN"]
        for i in range(len(ref_steps) - 1):
            from_tables, join_clause = self.join_tables(from_tables, join_clause, \
                                                        ref_steps[i + 1]["FROM"], ref_steps[i + 1]["JOIN"])
        order_by, order, having, group, discard, intersect, union = ([], [], [], [], [], [], [])
        # distinct = False
        for i in range(len(ref_steps)):
            group += ref_steps[i]["GROUP"]
            having += ref_steps[i]["HAVING"]
            order_by += ref_steps[i]["ORDER BY"]
            order += ref_steps[i]["ORDER"]
            discard += ref_steps[i]["DISCARD"]
            intersect += ref_steps[i]["INTERSECT"]
            union += ref_steps[i]["UNION"]
            # distinct = True if ref_steps[i]["distinct"] else distinct
        ref_conditions = [step["WHERE"] for step in ref_steps]
        select_cols = [col for select in ref_selects for col in select]
        return self.grounding_to_sql("union_column", \
                                     select_cols=list(OrderedDict.fromkeys(select_cols)), \
                                     from_tables=from_tables, \
                                     join_pairs=join_clause, \
                                     conditions=[col for where in ref_conditions for col in where], \
                                     group=list(dict.fromkeys(group)), \
                                     having=list(dict.fromkeys(having)), \
                                     order_by=list(dict.fromkeys(order_by)), \
                                     order=list(dict.fromkeys(order)), \
                                     union=list(dict.fromkeys(union)), \
                                     discard=list(dict.fromkeys(discard)), \
                                     intersect=list(dict.fromkeys(intersect)))

    def ground_discard_step(self, qdmr_step):
        refs = qdmr_step.arguments
        assert len(refs) == 2  # discard has 2 inputs
        input_orig, input_discarded = refs

        orig_step = self.grounded_steps[input_orig.replace("#", "")] if is_reference(input_orig) else \
            self.grounded_phrase_step(input_orig)  # orig is a column string: "instructors besides #2"
        discarded_step = self.grounded_steps[input_discarded.replace("#", "")] if is_reference(input_discarded) else \
            self.grounded_phrase_step(input_discarded)  # discarded is a column string: "#2 besides cats"

        assert (orig_step["SELECT"] == discarded_step["SELECT"] and
                len(orig_step["SELECT"]) == 1)
        return self.grounding_to_sql("discard", \
                                     select_cols=orig_step["SELECT"], \
                                     from_tables=orig_step["FROM"], \
                                     join_pairs=orig_step["JOIN"], \
                                     conditions=orig_step["WHERE"], \
                                     group=orig_step["GROUP"], \
                                     having=orig_step["HAVING"], \
                                     order_by=orig_step["ORDER BY"], \
                                     order=orig_step["ORDER"], \
                                     union=orig_step["UNION"], \
                                     discard=[(orig_step["SELECT"][0], discarded_step["SQL"])], \
                                     intersect=orig_step["INTERSECT"], \
                                     type_phrases=orig_step["type"])

    def ground_sort_step(self, qdmr_step):
        results, order_phrase = qdmr_step.arguments
        results = extract_refs(results)
        assert len(results) == 1
        # extract details of results step
        ref = results[0]
        results_step = self.grounded_steps[ref.replace("#", "")]
        # extract order steps
        order_refs = extract_refs(order_phrase)
        order_steps = [self.grounded_steps[r.replace("#", "")] for r in order_refs]
        # join the results step with the order steps
        from_tables, join_clause = results_step["FROM"], results_step["JOIN"]
        for step in order_steps:
            from_tables, join_clause = self.join_tables(from_tables, join_clause, \
                                                        step["FROM"], step["JOIN"])
        select_cols, group, having = \
            results_step["SELECT"], results_step["GROUP"], results_step["HAVING"]
        # check if there are order steps and extract order column
        order_selects = [step["SELECT"] for step in order_steps] if order_steps != [] \
            else [select_cols]
        order_by = list(itertools.chain.from_iterable(order_selects))
        if order_steps != [] and order_steps[0]["op"] == "group":
            select_cols += order_steps[0]["SELECT"]
            group += order_steps[0]["GROUP"]
        order_desc = "descend" in order_phrase or "decreas" in order_phrase
        return self.grounding_to_sql("sort", \
                                     select_cols=select_cols, \
                                     from_tables=list(dict.fromkeys(from_tables)), \
                                     join_pairs=list(dict.fromkeys(join_clause)), \
                                     conditions=results_step["WHERE"], \
                                     group=group, \
                                     having=having, \
                                     order_by=order_by, \
                                     order=["DESC"] if order_desc else ["ASC"], \
                                     type_phrases=results_step["type"])

    def ground_arithmetic_step(self, qdmr_step):
        arith, ref1, ref2 = qdmr_step.arguments
        arith_op = self.get_arithmetic_op(arith)
        assert arith_op is not None
        ref1 = ref1.replace("#", "")
        ref2 = ref2.replace("#", "")
        subquery1 = self.grounded_steps[ref1]["SQL"]
        subquery2 = self.grounded_steps[ref2]["SQL"]
        select_col = "(%s) %s (%s)" % (subquery1, arith_op, subquery2)
        return self.grounding_to_sql("arithmetic", \
                                     select_cols=[select_col])

    def get_arithmetic_op(self, arithmetic_phrase):
        op_map = {}
        op_map["sum"] = "+"
        op_map["difference"] = "-"
        op_map["multiplication"] = "*"
        op_map["division"] = "/"
        if arithmetic_phrase not in op_map.keys():
            return None
        return op_map[arithmetic_phrase]

    def schema_value_map(self):
        if self.query is None:
            return False

    def ground_value_in_column(self, phrase):
        """
        Go over phrase n-grams, return gold sql entities that match
        the largest overlapping n-gram of the phrase.
        Entities extracted from gold sql with their respective columns
        (string & number values)
        """
        val_col_list = []
        # check if phrase is an exact match with a schema value
        # if so, extract its relevant column
        normalized_phrase = " ".join(phrase.lower().replace("'", "").replace('"', "").split())
        normalized_phrase_tokens = normalized_phrase.split()
        value_to_column = sql_query_values(self.gold_query, self.schema)
        n = len(normalized_phrase_tokens)
        for i in reversed(range(1, n + 1)):
            # iterate all n-grams in decreasing order
            n_grams = [" ".join(tup) for tup in ngrams(normalized_phrase_tokens, i)]
            for value in value_to_column.keys():
                norm_value = value.replace("%", "")  # for LIKE '%%' values in query
                for n_gram in n_grams:
                    if ((n_gram in norm_value or norm_value in n_gram) \
                            and n_gram not in STOP_WORDS):
                        val_col_list += [(self.sql_value_case(value), value_to_column[value])]
            if len(val_col_list) > 0:
                # stop at largest n-gram value match
                return val_col_list
        return val_col_list

    def sql_value_case(self, value):
        """returns a non-numeric value in the case it appears in the
        original SQL query"""
        re_chars = ['(', ')', '[', ']', '^']
        for char in re_chars:
            # problematic chars in re patterns
            value = value.replace(char, "\%s" % char)
        # include starting parentheses to ensure it is a string value
        if re.search('"%s' % value, self.gold_query, re.IGNORECASE):
            return re.search('"%s' % value, self.gold_query, re.IGNORECASE).group(0)[1:]
        if re.search("'%s" % value, self.gold_query, re.IGNORECASE):
            return re.search("'%s" % value, self.gold_query, re.IGNORECASE).group(0)[1:]
        return re.search("%s" % value, self.gold_query, re.IGNORECASE).group(0)

    def match(self, phrase, schema_element):
        """check if phrase tokens are identical to
        schema_element/column tokens"""
        column = get_table_and_column(schema_element)[1] if ("." in schema_element) else None
        schema_element = schema_element.replace("_", " ").replace(".", " ")
        element_tokens = self.lemmatize(schema_element)
        phrase_tokens = self.lemmatize(phrase)
        if phrase_tokens == element_tokens:
            return True
        if len(phrase_tokens) > 1:
            # concatenated phrase heuristic,
            # e.g: "games played" --> sportsinfo.gamesplayed
            concat_phrase = phrase.replace(" ", "")
            element_name = schema_element if len(schema_element.split()) == 1 else schema_element.split()[1]
            if element_name == concat_phrase:
                return True
        if column and column not in ["id", "name"]:
            # check if phrase exactly matches a (non-trivial) column
            column = column.replace("_", " ")
            column_tokens = self.lemmatize(column)
            if phrase_tokens == column_tokens:
                return True
        return False

    def partial_match(self, phrase, schema_element):
        """check if ALL phrase tokens are contained in
        schema_element tokens"""
        if self.match(phrase, schema_element):
            return True
        schema_element = schema_element.replace("_", " ").replace(".", " ")
        element_tokens = schema_element.split()
        # lemmatize only the phrase for subsumption
        phrase_tokens = self.lemmatize(phrase)
        phrase_contained = True
        # check if all phrase tokens are contained in element tokens
        for tok in phrase_tokens:
            tok_contained = False
            for elem in element_tokens:
                if tok in elem or edit_distance(tok, elem) <= 1:
                    tok_contained = True
            phrase_contained = phrase_contained and tok_contained
        return phrase_contained

    def common_words_match(self, phrase, schema_element, reverse=False):
        """check if any of the phrase tokens appears in
        schema_element tokens"""
        if (self.match(phrase, schema_element) or
                self.partial_match(phrase, schema_element)):
            return True
        schema_element = schema_element.replace("_", " ").replace(".", " ")
        # lemmatize both phrase and element for common word match
        element_tokens = self.lemmatize(schema_element)
        phrase_tokens = self.lemmatize(phrase)
        stop_words = ['id', 'of', 'the', 'on'] + STOP_WORDS
        contained_tokens, tokens = (phrase_tokens, element_tokens) if not reverse else (element_tokens, phrase_tokens)
        for cont_tok in contained_tokens:
            for elem in tokens:
                if (cont_tok in elem) and (cont_tok not in stop_words):
                    return True
        return False

    def phrase_schema_candidates_NEW(self, phrase, type_constraint=None, type_phrases=None):
        """
        returns list of schema elements as phrase grounding candidates.
        candidates are ordered by their semantic similarity to the phrase.
        """
        schema_elements = self.schema.columns()
        schema_elements = self.schema.tables() + self.schema.columns()
        if type_constraint:
            # filter according to type
            assert (type_constraint in ["text", "number"])
            schema_elements = list(filter(lambda x: self.schema.column_type(x) == type_constraint, \
                                          schema_elements))
        # extend phrase subspan if necessary
        extended_phrase = type_phrases[0] if type_phrases else phrase
        # column match heuristic
        abbreviation_lexicon = {"first name": "fname", \
                                "last name": "lname", \
                                "source": "src", \
                                "destination": "dst", \
                                "gender": "sex", \
                                "airport": "ap", \
                                "paper": "publication title", \
                                "biggest state": "area", \
                                "largest state": "area", \
                                "smallest state": "area", \
                                "long": "length", \
                                "short": "length"}
        phrase = extended_phrase if phrase in ["name", "names", "last name", "last names", \
                                               "first name", "first names", "id", "ids", \
                                               "population"] else phrase 
        for token in abbreviation_lexicon.keys():
            extended_phrase = "%s %s" % (extended_phrase, abbreviation_lexicon[token]) if (token in phrase) \
                else extended_phrase
            # retrieve exact lexical matches
        scored_matches = self.average_scored_candidates(self.ranked_exact_matches(phrase, schema_elements), \
                                                        self.ranked_exact_matches(extended_phrase, schema_elements))
        # add the partial matches following the exact matches
        scored_matches += self.average_scored_candidates( \
            self.ranked_phrase_matches(phrase, schema_elements), \
            self.ranked_phrase_matches(extended_phrase, schema_elements))
        scored_matches += self.average_scored_candidates( \
            self.ranked_token_matches(phrase, schema_elements), \
            self.ranked_token_matches(extended_phrase, schema_elements))
        scored_matches += self.average_scored_candidates(self.semantic_ranked_candidates(phrase, schema_elements), \
                                                         self.semantic_ranked_candidates(extended_phrase,
                                                                                         schema_elements))
        # remove duplicates while preserving order
        scored_matches = list(OrderedDict.fromkeys(scored_matches))
        return scored_matches

    def average_scored_candidates(self, scored_candidates, other_scored_candidates):
        """Returns list of pairs of candidates and their averaged scores.
            List is sorted by the average scores."""
        if scored_candidates == []:
            return other_scored_candidates
        if other_scored_candidates == []:
            return scored_candidates
        score_dict = dict(scored_candidates)
        other_score_dict = dict(other_scored_candidates)
        avg_dictionary = {x: (score_dict.get(x, 0) + other_score_dict.get(x, 0)) / float(2)
                          for x in set(score_dict).union(other_score_dict)}
        return sorted(zip(avg_dictionary.keys(), avg_dictionary.values()), key=lambda x: x[1], reverse=True)

    def schema_match_candidate(self, phrase, type_constraint=None, type_phrases=None):
        """return a single schema element as the phrase grounding"""
        assert (type_constraint in ["text", "number"] \
                or type_constraint is None)
        schema_elements = self.schema.columns()
        schema_elements = self.schema.tables() + self.schema.columns()
        if type_constraint:
            # filter according to type
            schema_elements = list(filter(lambda x: self.schema.column_type(x) == type_constraint, \
                                          schema_elements))
        orig_phrase = phrase
        phrase_matches = [self.match(phrase, elem) for elem in schema_elements]
        match = True in phrase_matches
        phrase = type_phrases[0] if (type_phrases and not match) else phrase
        if "name" in phrase:
            # first/last name column match heuristic
            phrase = "%s fname" % phrase if ("first" in phrase) else phrase
            phrase = "%s lname" % phrase if ("last" in phrase) else phrase
        lex_cands_orig_phrase = self.lexical_match_candidate(orig_phrase, schema_elements)
        cand_score_orig = [(cand, self.semantic_match_score(orig_phrase, cand)) \
                           for cand in lex_cands_orig_phrase]
        cand_score_orig.sort(key=lambda x: x[1], reverse=True)
        lexical_candidates = self.lexical_match_candidate(phrase, schema_elements)
        candidate_to_score = [(cand, self.semantic_match_score(phrase, cand)) \
                              for cand in lexical_candidates]
        candidate_to_score.sort(key=lambda x: x[1], reverse=True)
        full_candidates_to_scores = []
        candidates_orig = dict(cand_score_orig)
        candidates_new = dict(candidate_to_score)
        for cand in candidates_orig.keys():
            score = (float(candidates_orig[cand] + candidates_new[cand]) / 2) \
                if (cand in candidates_new.keys()) else float(candidates_orig[cand]) / 2
            full_candidates_to_scores += [(cand, score)]
        for cand in candidates_new.keys():
            score = (float(candidates_orig[cand] + candidates_new[cand]) / 2) \
                if (cand in candidates_orig.keys()) else float(candidates_new[cand]) / 2
            full_candidates_to_scores += [(cand, score)]
        full_candidates_to_scores = list(dict.fromkeys(full_candidates_to_scores))
        full_candidates_to_scores.sort(key=lambda x: x[1], reverse=True)
        if len(lexical_candidates) > 1:
            if len(lexical_candidates) == 1:
                return lexical_candidates[0]
            else:
                return self.semantic_match_candidate(phrase, lexical_candidates)
        # expand to find phrase token match
        tokens = phrase.split(" ")
        for tok in tokens:
            lexical_candidates += self.lexical_match_candidate(tok, schema_elements)
        if len(lexical_candidates) == 1:
            return lexical_candidates[0]
        # last, return best candidate based on semantic similarity
        if len(lexical_candidates) > 1:
            return self.semantic_match_candidate(phrase, lexical_candidates)
        return self.semantic_match_candidate(phrase, schema_elements)

    def ranked_exact_matches(self, phrase, schema_elements):
        exact_matches = []
        # iterate over all schema elements and compute match
        for element in schema_elements:
            exact_matches = exact_matches + [element] if self.match(phrase, element) else exact_matches
        return self.semantic_ranked_candidates(phrase, exact_matches)

    def ranked_phrase_matches(self, phrase, schema_elements):
        phrase_matches = []
        # iterate over all schema elements and compute match
        for element in schema_elements:
            phrase_matches = phrase_matches + [element] \
                if self.partial_match(phrase, element) else phrase_matches
        return self.semantic_ranked_candidates(phrase, phrase_matches)

    def ranked_token_matches(self, phrase, schema_elements):
        token_matches = []
        # iterate over all schema elements and compute match
        for element in schema_elements:
            token_matches = token_matches + [element] \
                if self.common_words_match(phrase, element) \
                   or self.common_words_match(phrase, element, reverse=True) \
                else token_matches
        return self.semantic_ranked_candidates(phrase, token_matches)

    def lexical_match_candidate(self, phrase, schema_elements):
        phrase_matches = []
        token_matches = []
        # iterate over all schema elements and compute match
        for element in schema_elements:
            if self.match(phrase, element):  # exact match
                return [element]
            phrase_matches = phrase_matches + [element] \
                if self.partial_match(phrase, element) else phrase_matches
            token_matches = token_matches + [element] \
                if self.common_words_match(phrase, element) \
                   or self.common_words_match(element, phrase) \
                else token_matches
        if len(phrase_matches) > 0:
            return phrase_matches
        return token_matches

    def semantic_match_candidate(self, phrase, schema_elements):
        matches = []
        # iterate over all schema elements and compute similarity
        for element in schema_elements:
            score = self.semantic_match_score(phrase, element)
            matches += [(element, score)]
        sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
        # return highest scoring schema element
        best_match = sorted_matches[0][0]
        return best_match

    def semantic_match_score(self, phrase, schema_element):
        schema_element = schema_element.replace("_", " ").replace(".", " ")
        # lemmatized version:
        schema_element = ' '.join(self.lemmatize(schema_element))
        phrase = ' '.join(self.lemmatize(phrase))
        # remove duplicate following words, e.g., "film film id"
        schema_element = remove_duplicate_words(schema_element)
        return glove_sim(phrase, schema_element)

    def semantic_ranked_candidates(self, phrase, candidate_elements):
        scored_candidates = [(elem, self.semantic_match_score(phrase, elem)) for elem in candidate_elements]
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    def lemmatize(self, phrase):
        phrase = phrase.lower()
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(x) for x in wordninja.split(phrase)]
        stop_words = ['of', 'the']
        return list(filter(lambda x: x not in stop_words, tokens))


def extract_comparator(condition):
    """
    Returns comparator and value of a
    QDMR comparative step condition

    Parameters
    ----------
    condition : str
        Phrase representing condition of a QDMR step

    Returns
    -------
    tuple
        (comparator, value)
    """
    # extract comparative
    numbers = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", \
               "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    comparatives = {}
    comparatives["BETWEEN"] = ["between"]
    comparatives[">"] = ["more than", "above", "larger than", "larger", \
                         "older than", "older", "higher than", "higher", \
                         "greater than", "greater", "bigger than", "bigger", \
                         "after", "over"]
    comparatives[">="] = ["at least"]
    comparatives["<"] = ["less than", "under", "lower than", "lower", \
                         "younger than", "younger", "before", "below", \
                         "shorter than", "smaller than", "smaller"]
    comparatives["<="] = ["at most"]
    comparatives["!="] = ["is not"]
    comparatives["start"] = ['start with', 'starts with', 'begin']
    comparatives["end"] = ['end with', 'ends with']
    comparatives["LIKE"] = ["the letter", "the string", "the word", "the phrase", \
                            "contain", "include", "has", "have", \
                            "contains", "substring", "includes"]
    comparatives["="] = ['is equal to', 'equal to', 'same as', \
                         'is ', 'are ', 'was ']
    unformatted = {}
    unformatted[">="] = ["or later", "or more", "or after"]
    unformatted["<="] = ["or earlier", "or less", "or before"]
    ###TODO: handle "NOT LIKE"
    comp = None
    for c in comparatives.keys():
        if comp:
            break
        for trigger in comparatives[c]:
            if trigger in condition:
                comp = c
                break
    if comp:
        # extract value/reference
        value_phrase = condition.split(trigger)[1].strip()
        if comp == "BETWEEN":
            # "between num1 AND num2"
            return (comp, value_phrase.upper())
        elif comp:
            # check for unformatted comparators in value phrase
            for c in unformatted.keys():
                for trigger in unformatted[c]:
                    if trigger in condition:
                        comp = c
                        value_phrase = condition.split(trigger)[0].strip()
                        break
        for tok in value_phrase.split():
            if tok.isnumeric():
                return (comp, tok)
            if tok in numbers.keys():
                return (comp, numbers[tok])
        return (comp, value_phrase)
    return ("=", None)