from csv import DictWriter
from collections import OrderedDict

from sql_execution import *
from preprocess_db import *
from grounding_repairs import *


def handle_imdb_cast_table(sql):
    """cast is a reserved word in SQL.
    We alias the table cast if it appears in the query"""
    if "cast" not in sql:
        return sql
    sql = sql.replace(" cast.", " @@@placeholder@@@")
    sql = sql.replace("(cast.", "(@@@placeholder@@@")
    sql = sql.replace(" cast", " cast as cast_0")
    sql = sql.replace("@@@placeholder@@@", "cast_0.")
    return sql


def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='', encoding='utf-8') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)


def mixs(val):
    """""
    comparator function for sorting mixed lists of number & string
    """
    if isinstance(val, str):
        return (1, val, '')
    return (0, val, '')


class GroundingTestExample:
    def __init__(self, example_id, db_id, question, qdmr, gold_sql, schema_path, dataset):
        self.example_id = example_id
        self.db_id = db_id
        self.question = question
        self.qdmr = qdmr
        self.gold_sql = gold_sql
        default_path = "data/spider_databases/%s/%s.sqlite" % (self.db_id, self.db_id) \
            if dataset == "spider" else "data/other_databases/%s/%s.sqlite" % (self.db_id, self.db_id)
        self.schema_path = default_path if schema_path is None else schema_path
        self.dataset = dataset
        self.grounding = None
        self.grounded_sql = None
        self.grounding_error = False

    def to_dict(self):
        d = {}
        d["id"] = self.example_id
        d["db"] = self.db_id
        d["qdmr_grounding"] = self.qdmr
        d["gold_sql"] = self.gold_sql
        d["grounding"] = self.grounding.to_dict()
        return d

    def ground_example(self, assignment=None):
        if assignment:
            assert self.grounding
            self.grounding.assign_groundings(assignment)
        else:
            schema = prepare_db_schema(self.schema_path, self.dataset)
            self.grounding = QDMRGrounding(self.qdmr, self.question, schema, self.gold_sql)
        grounded_steps = self.grounding.ground()
        self.grounded_sql = self.grounding.get_grounded_sql()
        return self.grounded_sql is not None

    def set_qdmr(self, new_qdmr):
        self.qdmr = new_qdmr
        return True

    def set_grounded_sql(self, new_sql):
        self.grounded_sql = new_sql
        return True

    def get_gold_sql_denotation(self):
        return execute_sql(self.schema_path, self.gold_sql)

    def get_grounded_sql_denotation(self):
        if self.grounded_sql:
            sql = handle_imdb_cast_table(self.grounded_sql)
            try:
                denotation = execute_sql(self.schema_path, sql)
            except TimeoutError:
                print('* Grounded SQL execution timeout')
                denotation = None
            except (sqlite3.Warning, sqlite3.Error, sqlite3.DatabaseError,
                    sqlite3.IntegrityError, sqlite3.ProgrammingError,
                    sqlite3.OperationalError, sqlite3.NotSupportedError) as e:
                print('* Grounded SQL execution error')
                denotation = None
        return denotation

    def normalize_tuple(self, tup):
        # cast all tuple values to strings
        norm_vars = [str(var) for var in tup]
        return tuple(norm_vars)

    def normalize_denotation(self, denotation_list, distinct=None):
        if not denotation_list:
            return denotation_list
        # remove duplicates
        denotation_list = list(OrderedDict.fromkeys(denotation_list)) if distinct else denotation_list
        sort_tuples = [sorted(self.normalize_tuple(tup)) for tup in denotation_list]
        return sorted(sort_tuples)  # sort results set

    def correct_denotation(self, distinct=None):
        if not self.grounded_sql:
            return False
        if self.get_gold_sql_denotation() == self.get_grounded_sql_denotation():
            # unnormalized denotations
            return True
        gold_denotation_norm = self.normalize_denotation(self.get_gold_sql_denotation(), distinct=distinct)
        ground_denotation_norm = self.normalize_denotation(self.get_grounded_sql_denotation(), distinct=distinct)
        return gold_denotation_norm == ground_denotation_norm

    def repair(self, modules=None):
        """
        Run all repair modules for an incorrect grounding (wrong denotation).
        If nor correct grounding found return the original grounding.
        All repairs rely on the gold denotation for filtering.
        The modules input indicate which repairs should be run.

        Returns
        -------
        bool
            True if repair was successful, otherwise False
        """
        valid_modules = ["syntax", "column_ground", "qdmr"]
        if modules is not None:
            assert set(modules) <= set(valid_modules)
        assert self.grounding is not None
        original_grounding = self.grounding
        original_grounded_sql = self.grounded_sql
        repair_modules = valid_modules if modules is None else modules
        if "syntax" in repair_modules and self.syntax_repairs():
            return True
        self.restore_grounding(original_grounding, original_grounded_sql)
        if "column_ground" in repair_modules:
            repair = ColumnGroundingRepair(self)
            if repair.repair():
                return True
        self.restore_grounding(original_grounding, original_grounded_sql)
        if "qdmr" in repair_modules:
            repair = CountSumGroundingRepair(self)
            if repair.repair():
                return True
            repair = SuperlativeGroundingRepair(self)
            if repair.repair():
                return True
        self.restore_grounding(original_grounding, original_grounded_sql)
        print("*** Grounding example repair failed - restoring original grounding")
        return False

    def restore_grounding(self, grounding, grounded_sql):
        self.grounding = grounding
        self.grounded_sql = grounded_sql
        return True

    def syntax_repairs(self):
        """runs all sytax repairs on grounding example"""
        repair_count_distinct = AggrDistinctGroundingRepair(self)
        repair_like_val = LikeEqualsGroundingRepair(self)
        if repair_count_distinct.repair():
            return True
        return repair_like_val.repair()
