# Encode Grounded QDMR
# For each qdmr steps:
# 1. identify its operator
# 2. identify its qdmr arguments
# 3. ground *specific* arguments to columns, conditions, values in the DB

import json

from qdmr_encoding import encode_qdmr_steps, no_reference_encoding, has_reference_encoding
from qdmr_identifier import *
from tqdm import tqdm


def load_json(filepath):
    with open(filepath, "r") as reader:
        text = reader.read()
    return json.loads(text)


def write_to_json(data, json_file):
    with open(json_file, mode='w+', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    return True


def encoded_grounded_qdmr(grounded_qdmr_file, out_file):
    raw_data = load_json(grounded_qdmr_file)
    examples = raw_data["data"]
    enc_data = {}
    enc_data["data"] = []
    for i in tqdm(range(len(examples)), desc="Loading...", ascii=False, ncols=75):
        example = examples[i]
        # skip incorrectly grounded examples
        if (example["correct_denotation"] is False or
                example["grounding"] is None):
            continue
        ex_id = example["example_id"]
        db_id = example["db_id"]
        question = example["question"]
        qdmr = example["grounding"]["qdmr_grounding"]
        builder = QDMRProgramBuilder(qdmr)
        builder.build()
        steps = builder.steps
        grounded_steps = example["grounding"]["grounded_steps"]
        encoded_list = encode_qdmr_steps(steps, grounded_steps)
        enc_example = example
        enc_example["grounding_enc_no_ref"] = no_reference_encoding(encoded_list)
        enc_example["grounding_enc_has_ref"] = has_reference_encoding(encoded_list)
        enc_data["data"] += [enc_example]
    write_to_json(enc_data, out_file)
    num_examples = len(enc_data["data"])
    print(f"Done writing {num_examples} examples to {out_file}.")
    return True