import pandas as pd
from typing import Dict
import random
from tqdm import tqdm
import json
from wrapt_timeout_decorator import *
from ground_example import *

REPAIR_TIMEOUT = 600 # 10 minutes
REPAIR_TIMEOUT = 3000 # 50 minutes for Academic, IMDB & Yelp


def load_grounding_examples(examples_csv):
    """gets csv with examples to test"""
    data = pd.read_csv(examples_csv)
    return data


def read_grounding_example(example: Dict[str, str], schema_path):
    dataset = example['index'].split("_")[0].lower()
    return GroundingTestExample(example['index'], example['db_id'], example['question'],\
                                example['qdmr'], example['query'], schema_path, dataset)

# @timeout(dec_timeout=REPAIR_TIMEOUT, use_signals=False)
def try_repair(grounding_test):
    repair_res = grounding_test.repair()
    return repair_res

def write_one_grounding(example: Dict[str, str], output_file, field_names,\
                        json_file=None, schema_path=None):
    grounding_test = read_grounding_example(example, schema_path)
    row_dict = {}
    row_dict['example_id'] = grounding_test.example_id
    row_dict['db_id'] = grounding_test.db_id
    row_dict['question'] = grounding_test.question
    row_dict['qdmr'] = grounding_test.qdmr
    row_dict['sql_gold'] = grounding_test.gold_sql
    gold_denotation = grounding_test.get_gold_sql_denotation()
    row_dict['denotation_gold'] = gold_denotation[:min(200, len(gold_denotation))] if gold_denotation \
                                    else gold_denotation
    try:
        grounding_test.ground_example()
        row_dict['sql_ground'] = grounding_test.grounded_sql 
    except:
        grounding_test.grounding_error = True
        row_dict['sql_ground'] = row_dict['denotation_ground'] = "ERROR"
        row_dict['correct_denotation'] = False
    try:
        # grounded SQL execution terminates
        if not grounding_test.grounding_error:
            grounded_denotation = grounding_test.get_grounded_sql_denotation()
            row_dict['denotation_ground'] = grounded_denotation[:min(200, len(grounded_denotation))]
            row_dict['correct_denotation'] = grounding_test.correct_denotation(distinct=True)
    except:
        row_dict['denotation_ground'] = "ERROR"
        row_dict['correct_denotation'] = False

        #### try to repair grounding example
    if not row_dict['correct_denotation']:
        # try to repair the grounding
        try:
            if try_repair(grounding_test):
                repaired_sql = grounding_test.grounded_sql
                row_dict['correct_denotation'] = True
                row_dict['sql_ground'] = grounding_test.grounded_sql
                grounded_denotation = grounding_test.get_grounded_sql_denotation()
                row_dict['denotation_ground'] = grounded_denotation[:min(200, len(grounded_denotation))]
        except:
            row_dict['denotation_ground'] = "ERROR"
            row_dict['correct_denotation'] = False
    # write results
    append_dict_as_row(output_file, row_dict, field_names)
    if json_file:
        # full grounding data
        grounding_dict = None if row_dict['denotation_ground'] == "ERROR" else grounding_test.to_dict()
        row_dict['grounding'] = None if grounding_dict is None else grounding_dict['grounding']
        update_grounding_json(json_file, row_dict)
    return True

def write_to_json(data, json_file):
    with open(json_file, mode='w+', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    return True

def init_grounding_json(json_file):
    data = {}
    data['data'] = []
    return write_to_json(data, json_file)

def update_grounding_json(json_file, grounding_dict):
    with open(json_file, mode='r', encoding='utf-8') as file:
        data = json.load(file)
        temp = data['data']
        temp.append(grounding_dict)
    return write_to_json(data, json_file)

# @exit_after(180)
def write_grounding_results(grounding_examples, output_file, to_json=None):
    json_file = to_json
    if to_json:
        file = output_file.split(".")[0] if "." in output_file else output_file
        json_file = f"{file}.json"
        init_grounding_json(json_file)
    with open(output_file, mode='w', encoding='utf-8') as csv_file:
        field_names = ['example_id','db_id','question','qdmr',\
                       'sql_gold', 'sql_ground', 'denotation_gold', 'denotation_ground',\
                       'correct_denotation']
        writer = DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
    n = 6955
    # random_ints = [i for i in range(n)][:500] #### Spider train split
    # random_ints = [i for i in range(6955, 7136)] #### Academic split 
    # random_ints = [i for i in range(7136, 7384)] #### Geo split
    # random_ints = [i for i in range(7384, 11426)] #### ATIS split 
    # random_ints = [i for i in range(11426, 11557)] #### IMDB split
    # random_ints = [i for i in range(11557, 11685)] #### Yelp split
    # random_ints = [i for i in range(11426, 11685)] #### IMDB+Yelp split
    # random_ints = [i for i in range(11685, 12562)] #### Geo880 split
    # random_ints = [i for i in range(11685, 11697)] #### Geo880 split
    # random_ints = [i for i in range(12562, 13589)] #### Spider dev split
    # random_ints = [i for i in range(13589)] #### Entire example split
    # random_ints = [i for i in range(11685, 13589)]  #### Geo880 + Spider dev split
    random_ints = [i for i in range(len(grounding_examples))] # use all file examples
    # random.shuffle(random_ints)
    for i in tqdm(range(len(random_ints)), desc="Loading...", ascii=False, ncols=75):
        try:
            write_one_grounding(grounding_examples.iloc[random_ints[i]], output_file, \
                                field_names=field_names, json_file=json_file)
        except:
            continue
    # write_one_grounding(grounding_examples.iloc[12576], output_file,\
    #                     field_names=field_names, json_file=json_file)
    print("Complete.")
    return True


