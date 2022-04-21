from eval_exec.db_schema import DBSchema
from eval_exec.grounded_qdmr import GroundedQDMR
from eval_exec.utils import get_table_and_column
from eval_exec.preprocess_db import prepare_db_schema
import re


def qdmr_to_sql(qdmr_ref_steps_encoding, question, db_schema_path, dataset=None):
    dataset = "spider" if dataset in ["spider", None] else dataset
    schema = prepare_db_schema(db_schema_path, dataset=dataset)
    grounded_qdmr = GroundedQDMR(qdmr_ref_steps_encoding, question, schema)
    grounded_qdmr.to_sql()
    n = str(len(grounded_qdmr.sql_steps))
    return grounded_qdmr.sql_steps[n]["SQL"]


def prune_nested_queries(pred_sql, schema_path, lowercased=None):
    schema = DBSchema(schema_path)
    columns = schema.columns()
    for col in columns:
        table, _ = get_table_and_column(col)
        redundant_nested_cond = f"AND {col} IN ( SELECT {col} FROM {table} )"
        redundant_nested_cond = redundant_nested_cond.lower() if lowercased else redundant_nested_cond
        if redundant_nested_cond in pred_sql:
            pred_sql = pred_sql.replace(redundant_nested_cond, "")
    return pred_sql


def normalize_qdmr_prediction(predicted_qdmr):
    def fix_not_equal(pred):
        op = "!="
        if op in pred and " " + op not in pred:
            return pred.replace(op, " " + op)
        return pred

    def replace_trailing_dot(pred):
        if " ." in pred:
            pred = pred.replace(" .", ".")
        if ". " in pred:
            pred = pred.replace(". ", ".")
        return pred

    def handle_like_value(pred):
        def add_like_pct(string):
            extracted_value = string.split("like ")[1][:-1].strip()  # omit last ")"
            pct_value = f"%{extracted_value}%"
            return string.replace(extracted_value, pct_value)

        like_cond_val = [item.group(0) for item in re.finditer(r'like .*? \)', pred)]
        for cond in like_cond_val:
            pred = pred.replace(cond, add_like_pct(cond))
        return pred

    return handle_like_value(replace_trailing_dot(fix_not_equal(predicted_qdmr)))
