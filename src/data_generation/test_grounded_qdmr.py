# TODO: ensure the mapping to SQL of grounded QDMRs is accurate
#   1. Read all grounded QDMRs along with their SQL queries for: Spider train, dev & Geo
#   2. Map grounded QDMRs to SQL
#   3. Compare execution results of grounded SQL versus mapped SQL and debug errors
#       3.1. Potential self-join issues
#   4. Deal with the conversion of parentheses QDMR encoding to reference-based encoding
import random

from tqdm import tqdm
from db_schema import DBSchema
from grounded_qdmr import GroundedQDMR
from preprocess_db import prepare_db_schema
from qdmr_encoding_parser import formula_to_ref_encoding
from sql_execution import correct_denotation
from write_encoding import encoded_grounded_qdmr, load_json, write_to_json


# 0. Create the grounded QDMR encodings

def create_qdmr_encodings():
    # grounded_qdmr_file = "data/qdmr_grounding/qdmr_groundings_unlimited_spider_train.json"
    # encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/qdmr_ground_enc_spider_train.json")
    # grounded_qdmr_file = "data/qdmr_grounding/groundings_spider_dev.json"
    # encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/qdmr_ground_enc_spider_dev.json")
    # grounded_qdmr_file = "data/qdmr_grounding/groundings_geo880.json"
    # encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/qdmr_ground_enc_geo880.json")
    # grounded_qdmr_file = "data/qdmr_grounding/groundings_predicted_spider_dev.json"
    # encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/qdmr_ground_enc_predicted_spider_dev.json")
    # grounded_qdmr_file = "data/qdmr_grounding/groundings_predicted_spider_train_40_db.json"
    # encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/qdmr_ground_enc_predicted_spider_train_40_db.json")
    # grounded_qdmr_file = "data/qdmr_grounding/groundings_predicted_spider_train_40_db_02_V2.json"
    # encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/qdmr_ground_enc_predicted_spider_train_40_db_V2.json")
    # grounded_qdmr_file = "data/qdmr_grounding/groundings_predicted_spider_train_30_db_02.json"
    # encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/qdmr_ground_enc_predicted_spider_train_30_db.json")
    # grounded_qdmr_file = "data/qdmr_grounding/groundings_predicted_geo880_train_zero_shot.json"
    # encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/qdmr_ground_enc_predicted_geo_train_zero_shot.json")
    grounded_qdmr_file = "data/qdmr_grounding/spider_train_few_shot_groundings/groundings_predicted_spider_train_few_shot_full.json"
    encoded_grounded_qdmr(grounded_qdmr_file, "data/qdmr_grounding/spider_train_few_shot_groundings/qdmr_ground_enc_predicted_spider_train_few_shot.json")


# 1. Convert map encoding to SQL and compare execution with grounded SQL

def encoding_to_sql(has_ref_encoding, question, db_dir, db_name, dataset=None):
    schema_path = "%s/%s/%s.sqlite" % (db_dir, db_name, db_name)
    # add missing join paths for non-Spider DB schema
    dataset = "spider" if dataset in ["spider", None] else dataset
    schema = prepare_db_schema(schema_path, dataset=dataset)
    grounded_qdmr = GroundedQDMR(has_ref_encoding, question, schema)
    grounded_qdmr.to_sql()
    n = str(len(grounded_qdmr.sql_steps))
    return grounded_qdmr.sql_steps[n]["SQL"]


def evaluate_grounded_qdmr(grounded_qdmr_file, db_dir, output_file, encoding=None, dataset=None):
    assert encoding in ["no_ref", "has_ref", None]
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
        enc_example["qdmr"] = example["grounding"]["qdmr_grounding"]
        enc_example["sql_ground"] = example["grounding"]["grounded_sql"]
        enc_example["qdmr_ground_enc"] = example["grounding_enc_has_ref"]
        if encoding == "no_ref":
            formula_encoding = example["grounding_enc_no_ref"]
            enc_example["qdmr_ground_enc_original"] = formula_encoding
            try:
                enc_example["qdmr_ground_enc"] = formula_to_ref_encoding(formula_encoding)
            except:
                enc_example["qdmr_ground_enc"] = "CONVERSION_ERROR"
        try:
            enc_example["sql_enc"] = encoding_to_sql(enc_example["qdmr_ground_enc"], enc_example["question"],
                                                     db_dir, enc_example["db_name"], dataset=dataset)
        except:
            enc_example["sql_enc"] = "ERROR"
        db_path = "%s/%s/%s.sqlite" % (db_dir, enc_example["db_name"], enc_example["db_name"])
        if enc_example["sql_enc"] == "ERROR":
            denotation_flag = False
        else:
            denotation_flag = correct_denotation(enc_example["sql_enc"], enc_example["sql_ground"], db_path,
                                                 distinct=None)
        enc_example["correct_enc_denotation"] = denotation_flag
        num_correct = num_correct + 1 if denotation_flag else num_correct
        enc_data["data"] += [enc_example]
    write_to_json(enc_data, output_file)
    num_examples = len(enc_data["data"])
    print(f"Done writing {num_examples} examples to {output_file}.")
    print(f"Number of correct grounded enc. denotations: {num_correct}/{num_examples}.")
    return True


create_qdmr_encodings()

# Spider dataset
# evaluate_grounded_qdmr(grounded_qdmr_file="data/qdmr_grounding/qdmr_ground_enc_predicted_spider_train_30_db.json",
#                        db_dir="data/spider_databases",
#                        output_file="data/qdmr_grounding/test_encoding_to_sql_02.json",
#                        encoding="no_ref")

# Geo880 dataset
# evaluate_grounded_qdmr(grounded_qdmr_file="data/qdmr_grounding/qdmr_ground_enc_geo880.json",
#                        db_dir="data/other_databases",
#                        output_file="data/qdmr_grounding/test_encoding_to_sql_geo880.json",
#                        encoding="no_ref",
#                        dataset="geo")

# Test why the encoding-to-SQL mapping returns different results between runs - join_path_chain

# path1 = "data/qdmr_grounding/test_encoding_to_sql_01.json"
# path2 = "data/qdmr_grounding/test_encoding_to_sql_02.json"
#
# data = load_json(path1)
# other_data = load_json(path2)
# examples = data["data"]
# other_examples = other_data["data"]
#
# for i in range(len(examples)):
#     example = examples[i]
#     other_example = other_examples[i]
#     if example["correct_enc_denotation"] != other_example["correct_enc_denotation"]:
#         print("* example: ", example["ex_id"])
#         print("* grounded: ", example["sql_ground"])
#         print("* qdmr encoded SQL 01: ", example["sql_enc"])
#         print("* result: ", example["correct_enc_denotation"])
#         print("* qdmr encoded SQL 02: ", other_example["sql_enc"])
#         print("* result: ", other_example["correct_enc_denotation"])
#         print("*"*20)


def test_unite_two_qdmr_jsons(json_file, other_json_file, dataset_name, output_json):
    raw_data = load_json(json_file)
    examples = raw_data["data"]
    other_raw_data = load_json(other_json_file)
    other_examples = other_raw_data["data"]
    all_examples = examples + other_examples
    dataset_examples = []
    for ex in all_examples:
        if dataset_name in ex["example_id"]:
            dataset_examples += [ex]
    enc_data = {"data": dataset_examples}
    write_to_json(enc_data, output_json)
    num_examples = len(enc_data["data"])
    print(f"Done writing {num_examples} examples to {output_json}.")
    return True


def test_spider_random_sample(spider_json, output_json, sample_size):
    raw_data = load_json(spider_json)
    sample = list(random.sample(raw_data, sample_size))
    write_to_json(sample, output_json)
    num_examples = len(sample)
    print(f"Done writing {num_examples} examples to {output_json}.")
    for ex in sample:
        print(ex["query"])
    return True


# test_spider_random_sample(spider_json="data/spider_queries/dev.json",
#                           output_json="test_sample_spider_dev.json",
#                           sample_size=100)
