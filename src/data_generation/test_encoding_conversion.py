# TODO: Test the mapping of grounded QDMR encoded as formula to the reference steps encoding
#   E.g.:
#   project ( table2.column2 , select ( table.column ) ) -->
#       select ( table.column ) ; project ( table2.column2 , #1 )
#   Steps:
#   1. Read grounded QDMR encodings from file
#   2. Convert formula (no-ref) encodings to ref steps encoding
#   3. Compare the converted encoding to the original ref steps encoding
#   4. Print error if the converted ref steps is different from the original
#   5. Check if the different ref step encodings (original & converted) are still equivalent

from tqdm import tqdm

from qdmr_encoding_parser import formula_to_ref_encoding
from write_encoding import load_json, write_to_json


def test_enc_conversion(grounded_qdmr_file, output_file):
    raw_data = load_json(grounded_qdmr_file)
    examples = raw_data["data"]
    failed_examples = {}
    failed_examples["data"] = []
    num_correct = 0
    for i in tqdm(range(len(examples)), desc="Loading...", ascii=False, ncols=75):
        example = examples[i]
        enc_example = {}
        enc_example["ex_id"] = example["example_id"]
        enc_example["db_name"] = example["db_id"]
        enc_example["question"] = example["question"]
        enc_example["qdmr"] = example["grounding"]["qdmr_grounding"]
        enc_example["sql_ground"] = example["grounding"]["grounded_sql"]
        enc_example["qdmr_ref_enc"] = example["grounding_enc_has_ref"]
        enc_example["qdmr_formula_enc"] = example["grounding_enc_no_ref"]
        enc_example["error"] = None
        try:
            enc_example["converted_ref_enc"] = formula_to_ref_encoding(enc_example["qdmr_formula_enc"])
        except:
            enc_example["converted_ref_enc"] = None
            enc_example["error"] = "PARSE_ERROR"
            failed_examples["data"] += [enc_example]
        finally:
            if enc_example["error"] is None:
                if enc_example["converted_ref_enc"] == enc_example["qdmr_ref_enc"]:
                    num_correct += 1
                else:
                    enc_example["error"] = "CONVERSION_ERROR"
                    failed_examples["data"] += [enc_example]
    write_to_json(failed_examples, output_file)
    num_examples = len(examples)
    print(f"Done writing {num_examples} examples to {output_file}.")
    print(f"Number of correctly converted formula encodings: {num_correct}/{num_examples}.")
    return True
