# TODO: test the execution accuracy of predicted grounded-QDMRs
#   1. Read all predicted grounded QDMRs along with their gold SQL queries for: Spider dev
#   2. Map grounded QDMRs to SQL
#   3. Adjust SQL value casing based on the gold SQL values
#   4. Compare execution results of gold SQL versus predicted QDMR
#   5. Compare the execution results ref-steps and formula encodings of grounded QDMR
import random
import re
from csv import DictWriter

from tqdm import tqdm

from db_schema import DBSchema
from ground_example import append_dict_as_row
from predicted_sql import fix_sql_casing
from qdmr_encoding_parser import formula_to_ref_encoding
from sql_execution import correct_denotation
from test_grounded_qdmr import encoding_to_sql
from utils import get_table_and_column
from write_encoding import load_json, write_to_json


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
        # pred = pred.replace("song_id", "singer_id").replace("conzert", "concert").replace("orchestre", "orchestra")#####TODO: delete this!!!!
        return pred

    return handle_like_value(replace_trailing_dot(fix_not_equal(predicted_qdmr)))


def evaluate_predicted_qdmr(grounded_qdmr_file, predicted_qdmr_file, db_dir, output_file, encoding=None):
    assert encoding in ["no_ref", "has_ref", None]
    with open(predicted_qdmr_file) as f:
        predictions = [pred.replace("\n", "") for pred in f.readlines()]
    raw_data = load_json(grounded_qdmr_file)
    examples = raw_data["data"]
    enc_data = {}
    enc_data["data"] = []
    num_correct = 0
    for i in tqdm(range(len(examples)), desc="Loading...", ascii=False, ncols=75):
        example = examples[i]
        enc_example = {}
        enc_example["ex_id"] = example["example_id"]
        enc_example["db_name"] = example["db_id"]
        enc_example["question"] = example["question"]
        enc_example["sql_gold"] = example["sql_gold"]
        enc_example["qdmr"] = example["grounding"]["qdmr_grounding"]
        enc_example["sql_ground"] = example["grounding"]["grounded_sql"]
        enc_example["qdmr_formula_enc"] = example["grounding_enc_no_ref"]
        enc_example["qdmr_ref_enc"] = example["grounding_enc_has_ref"]
        enc_example["predicted_qdmr"] = normalize_qdmr_prediction(predictions[i])
        pred_qdmr_ref_steps = enc_example["predicted_qdmr"]
        try:
            if encoding == "no_ref":
                enc_example["predicted_qdmr_mapped"] = formula_to_ref_encoding(enc_example["predicted_qdmr"])
                pred_qdmr_ref_steps = enc_example["predicted_qdmr_mapped"]
            pred_sql_uncased = encoding_to_sql(pred_qdmr_ref_steps, enc_example["question"],
                                               db_dir, enc_example["db_name"])
            enc_example["pred_sql"] = fix_sql_casing(pred_sql_uncased, enc_example["sql_gold"])
        except:
            enc_example["pred_sql"] = "ERROR"
        db_path = "%s/%s/%s.sqlite" % (db_dir, enc_example["db_name"], enc_example["db_name"])
        enc_example["pred_sql"] = prune_nested_queries(enc_example["pred_sql"], db_path)
        if enc_example["pred_sql"] == "ERROR":
            denotation_flag = False
        else:
            denotation_flag = correct_denotation(enc_example["pred_sql"], enc_example["sql_gold"], db_path,
                                                 distinct=True)
        enc_example["correct_enc_denotation"] = denotation_flag
        num_correct = num_correct + 1 if denotation_flag else num_correct
        enc_data["data"] += [enc_example]
    write_to_json(enc_data, output_file)
    num_examples = len(enc_data["data"])
    print(f"Done writing {num_examples} examples to {output_file}.")
    print(f"Number of correct predicted examples: {num_correct}/{num_examples}.")
    return True


def evaluate_predicted_qdmr_on_spider(spider_queries_file, predicted_qdmr_file,
                                      db_dir, output_file, encoding=None, write_csv=None):
    assert encoding in ["no_ref", "has_ref", None]
    with open(predicted_qdmr_file, encoding='utf-8') as f:
        predictions = [pred.replace("\n", "") for pred in f.readlines()]
    examples = load_json(spider_queries_file)
    enc_data = {}
    enc_data["data"] = []
    num_correct = 0
    field_names = ['ex_id', 'db_name', 'question', 'sql_gold', 'predicted_qdmr',
                   'predicted_qdmr_mapped', 'pred_sql', 'correct_enc_denotation']
    if write_csv:
        init_preds_csv(write_csv, field_names)
    for i in tqdm(range(len(examples)), desc="Loading...", ascii=False, ncols=75):
        example = examples[i]
        enc_example = {}
        enc_example["ex_id"] = i
        enc_example["db_name"] = example["db_id"]
        enc_example["question"] = example["question"]
        enc_example["sql_gold"] = example["query"]
        enc_example["predicted_qdmr"] = normalize_qdmr_prediction(predictions[i])
        pred_qdmr_ref_steps = enc_example["predicted_qdmr"]
        try:
            if encoding == "no_ref":
                enc_example["predicted_qdmr_mapped"] = formula_to_ref_encoding(enc_example["predicted_qdmr"])
                pred_qdmr_ref_steps = enc_example["predicted_qdmr_mapped"]
            pred_sql_uncased = encoding_to_sql(pred_qdmr_ref_steps, enc_example["question"],
                                               db_dir, enc_example["db_name"])
            enc_example["pred_sql"] = fix_sql_casing(pred_sql_uncased, enc_example["sql_gold"])
        except:
            enc_example["pred_sql"] = "ERROR"
        db_path = "%s/%s/%s.sqlite" % (db_dir, enc_example["db_name"], enc_example["db_name"])
        print("enc_example[pred_sql]: ", enc_example["pred_sql"])
        enc_example["pred_sql"] = prune_nested_queries(enc_example["pred_sql"], db_path)
        if enc_example["pred_sql"] == "ERROR":
            denotation_flag = False
        else:
            denotation_flag = correct_denotation(enc_example["pred_sql"], enc_example["sql_gold"], db_path,
                                                 distinct=True)
        enc_example["correct_enc_denotation"] = denotation_flag
        num_correct = num_correct + 1 if denotation_flag else num_correct
        enc_data["data"] += [enc_example]
        if write_csv:
            append_dict_as_row(write_csv, enc_example, field_names)
    write_to_json(enc_data, output_file)
    num_examples = len(enc_data["data"])
    print(f"Done writing {num_examples} examples to {output_file}.")
    print(f"Number of correct predicted examples: {num_correct}/{num_examples}.")
    return True


def evaluate_predicted_sql_on_spider(spider_queries_file, predicted_sql_file,
                                     db_dir, output_file, write_csv=None):
    with open(predicted_sql_file, encoding='utf-8') as f:
        predictions = [pred.replace("\n", "") for pred in f.readlines()]
    examples = load_json(spider_queries_file)
    pred_data = {}
    pred_data["data"] = []
    num_correct = 0
    field_names = ['ex_id', 'db_name', 'question', 'sql_gold', 'pred_sql', 'correct_denotation']
    if write_csv:
        init_preds_csv(write_csv, field_names)
    for i in tqdm(range(len(examples)), desc="Loading...", ascii=False, ncols=75):
        example = examples[i]
        pred_example = {}
        pred_example["ex_id"] = i
        pred_example["db_name"] = example["db_id"]
        pred_example["question"] = example["question"]
        pred_example["sql_gold"] = example["query"]
        pred_example["pred_sql"] = predictions[i]
        try:
            pred_example["pred_sql"] = fix_sql_casing(pred_example["pred_sql"], pred_example["sql_gold"])
        except:
            pred_example["pred_sql"] = "ERROR"
        db_path = "%s/%s/%s.sqlite" % (db_dir, pred_example["db_name"], pred_example["db_name"])
        # TODO: delete
        print("pred_example: ", pred_example)
        if pred_example["pred_sql"] == "ERROR":
            denotation_flag = False
        else:
            denotation_flag = correct_denotation(pred_example["pred_sql"], pred_example["sql_gold"], db_path,
                                                 distinct=True)
        pred_example["correct_denotation"] = denotation_flag
        num_correct = num_correct + 1 if denotation_flag else num_correct
        pred_data["data"] += [pred_example]
        if write_csv:
            append_dict_as_row(write_csv, pred_example, field_names)
    write_to_json(pred_data, output_file)
    num_examples = len(pred_data["data"])
    print(f"Done writing {num_examples} examples to {output_file}.")
    print(f"Number of correct predicted examples: {num_correct}/{num_examples}.")
    return True


def stupid_distinct_hack(sql, question):
    def contains_distinct_trigger(text):
        text = text.lower()
        triggers = ["distinct", "different", "unique"]
        for t in triggers:
            if t in text:
                return True
        return False

    def starts_with_select_operator(text, operator):
        prefix = f"select {operator}"
        if low_sql.startswith(prefix):
            return True
        return False

    low_sql = sql.lower()
    if (starts_with_select_operator(low_sql, "count") and not starts_with_select_operator(low_sql, "count(distinct")) \
            and contains_distinct_trigger(question):
        sql = sql.replace("SELECT count(", "SELECT count(DISTINCT ", 1)
        sql = sql.replace("SELECT COUNT(", "SELECT COUNT(DISTINCT ", 1)
        return sql
    aggregate_ops = ["min", "max", "sum", "avg", "distinct", "count"]
    for op in aggregate_ops:
        if starts_with_select_operator(low_sql, op):
            return sql
    if "order by" in low_sql and "limit" not in low_sql:
        return sql
    return sql.replace("SELECT ", "SELECT DISTINCT ", 1)


def init_preds_csv(output_csv, field_names):
    with open(output_csv, mode='w', encoding='utf-8') as csv_file:
        writer = DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()


# evaluate_predicted_qdmr(grounded_qdmr_file="data/qdmr_grounding/qdmr_ground_enc_spider_dev.json",
#                         predicted_qdmr_file="predicted_qdmr_ref_test01.txt",
#                         db_dir="data/spider_databases",
#                         output_file="data/qdmr_grounding/test_predicted_ref_encoding_to_sql.json",
#                         encoding="has_ref")

# evaluate_predicted_qdmr(grounded_qdmr_file="data/qdmr_grounding/qdmr_ground_enc_spider_dev.json",
#                         predicted_qdmr_file="predicted_qdmr_formula_ckpt55.txt",
#                         db_dir="data/spider_databases",
#                         output_file="data/qdmr_grounding/test_predicted_formula_encoding_to_sql.json",
#                         encoding="no_ref")

evaluate_predicted_qdmr_on_spider(spider_queries_file="data/spider_queries/dev.json",
                                  predicted_qdmr_file="data/model_predictions/pred_spider_dev_qdmr_formula_few_shot_700_seed_43_99ckpt.txt",
                                  db_dir="data/spider_databases",
                                  output_file="data/qdmr_grounding/test_predicted_formula_encoding_to_sql_full_spider_dev.json",
                                  encoding="no_ref",
                                  write_csv="data/qdmr_grounding/test_predicted_formula_encoding_to_sql_full_spider_dev_no_set.csv")

# evaluate_predicted_sql_on_spider(spider_queries_file="data/spider_queries/dev.json",
#                                  predicted_sql_file="data/model_predictions/t5_large_spider_partialgold_seed_42.txt",
#                                  db_dir="data/spider_databases",
#                                  output_file="data/qdmr_grounding/test_predicted_gold_sql_full_spider_dev.json",
#                                  write_csv="data/qdmr_grounding/test_predicted_gold_sql_full_spider_dev.csv")
