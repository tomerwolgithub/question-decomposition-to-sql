import json

from eval_spider import ExecutionAccuracy


def load_json_dup(filepath):
    with open(filepath, "r") as reader:
        text = reader.read()
    return json.loads(text)


def read_spider_examples(train_dataset_path):
    inputs = []
    # Read examples
    raw_data = load_json_dup(train_dataset_path)
    for example in raw_data:
        database = example["db_id"].lower()
        source = example["question"]
        target = example["query"]
        # Prepend database
        source = "%s: %s" % (database, source)
        input = source.lower()
        target = target.lower()
        inputs.append(input)
    return inputs


def read_lines_from_file_sql(sql_file):
    with open(sql_file, encoding='UTF-8') as f:
        glist = [l.strip().split('\t')[0] for l in f.readlines() if len(l.strip()) > 0]
    return glist


def prepare_qdmr_grounded_sql_preds(grounded_sql_path, gold_examples_length):
    """Read qdmr-grounded SQL and lowercase them to mimic T5 predictions.
    Order the predictions according to Spider's original order.
    Where no valid SQL was grounded place an empty string as the default value."""
    DEFAULT = ""
    raw_data = load_json_dup(grounded_sql_path)
    examples = raw_data["data"]
    ex_id_to_sql = {}
    for ex in examples:
        spider_id = int(ex["example_id"].split("_")[-1])
        correct_grounded_sql = ex["correct_denotation"]
        sql = ex["sql_ground"] if correct_grounded_sql else DEFAULT
        sql = sql.lower()
        ex_id_to_sql[spider_id] = sql
    ordered_sql = []
    for idx in range(gold_examples_length):
        if idx not in ex_id_to_sql:
            ex_id_to_sql[idx] = DEFAULT
        ordered_sql += [ex_id_to_sql[idx]]
    return ordered_sql


spider_train_path = "data/spider_queries/train_spider.json"
spider_dev_path = "data/spider_queries/dev.json"

train_examples = read_spider_examples(spider_train_path)
dev_examples = read_spider_examples(spider_dev_path)

gold_sql_dev = "data/spider_queries/dev_gold.sql"
gold_sql_list = read_lines_from_file_sql(gold_sql_dev)

pred_sql_dev = "data/predictions/t5_spider_bs_4_accum_32_epochs_100_lr_1e-4_checkpoint_95.txt"
pred_sql_list = read_lines_from_file_sql(pred_sql_dev)

# pred_qdmr_formula_dev = "data/predictions/preds_t5_spider_formula_data_aug_bs_4_accum_32_epochs_150_lr_1e-4_ckpt_62.txt"
pred_qdmr_formula_dev = "data/predictions/preds_t5_large_spider_formula_bs_1_accum_128_epochs_150_lr_1e-4_seed_42_ckpt_25.txt"
pred_qdmr_formula_list = read_lines_from_file_sql(pred_qdmr_formula_dev)

pred_qdmr_steps_dev = "data/predictions/preds_t5_spider_hasref_bs_4_accum_32_epochs_150_lr_1e-4_ckpt_63.txt"
pred_qdmr_steps_list = read_lines_from_file_sql(pred_qdmr_steps_dev)

# grounded_sql_preds = prepare_qdmr_grounded_sql_preds("data/predictions/groundings_spider_dev.json", len(gold_sql_list))

pred_grounded_sql_dev = "data/predictions/preds_t5_spider_groundsql_bs_4_accum_32_epochs_150_lr_1e-4_ckpt_144.txt"
grounded_sql_preds = read_lines_from_file_sql(pred_grounded_sql_dev)

pred_gold_sql_dev = "data/predictions/preds_t5_large_spider_gold_bs_1_accum_128_epochs_150_lr_1e-4_seed_42_ckpt_10.txt"
gold_sql_preds = read_lines_from_file_sql(pred_gold_sql_dev)

exec_accuracy = ExecutionAccuracy()
# x = exec_accuracy.evaluate(examples=dev_examples,
#                            gold=gold_sql_list,
#                            predict=gold_sql_preds,
#                            db_dir="data/spider_databases",
#                            exec_set_match=True)

# x = exec_accuracy.evaluate(examples=dev_examples,
#                            gold=gold_sql_list,
#                            predict=grounded_sql_preds,#pred_sql_list,
#                            db_dir="data/spider_databases",
#                            exec_set_match=True)

# x = exec_accuracy.evaluate(examples=dev_examples,
#                            gold=gold_sql_list,
#                            predict=pred_qdmr_formula_list,
#                            db_dir="data/spider_databases",
#                            exec_set_match=True,
#                            pred_type="qdmr_formula")

# x = exec_accuracy.evaluate(examples=dev_examples,
#                            gold=gold_sql_list,
#                            predict=pred_qdmr_steps_list,
#                            db_dir="data/spider_databases",
#                            exec_set_match=True,
#                            pred_type="qdmr_steps")


# GEO880 evaluation

geo_train_path = "data/geo_queries/geo880_sql_train.json"
geo_dev_path = "data/geo_queries/geo880_sql_dev.json"
geo_test_path = "data/geo_queries/geo880_sql_test.json"

gold_sql_dev = "data/geo_queries/dev_gold_geo.sql"
gold_sql_list_dev = read_lines_from_file_sql(gold_sql_dev)

gold_sql_test = "data/geo_queries/test_gold_geo.sql"
gold_sql_list_test = read_lines_from_file_sql(gold_sql_test)

pred_qdmr_formula_test = "data/predictions/geo/t5_base_geo880_qdmr_formula_preds.txt"
pred_qdmr_formula_list = read_lines_from_file_sql(pred_qdmr_formula_test)

dbs_list = ["geo:"] * len(gold_sql_list_test)
x = exec_accuracy.evaluate(examples=dbs_list,
                           gold=gold_sql_list_test,
                           predict=pred_qdmr_formula_list,
                           db_dir="data/geo_database",
                           exec_set_match=True,
                           pred_type="qdmr_formula",
                           dataset="geo")


print(x)

