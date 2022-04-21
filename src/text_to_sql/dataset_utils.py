import json

def load_json(filepath):
    with open(filepath, "r") as reader:
        text = reader.read()
    return json.loads(text)


def normalize_whitespace(source):
    tokens = source.split()
    return " ".join(tokens)
    
    
def write_to_json(data, json_file):
    with open(json_file, mode='w+', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    return True


def geo_data_splits(input_data, output_file, data_split):
    """
    script to generate Geo880 data file in the QDMRDataset json format:
        1. Geo880 gold SQL json files for train/dev/test split
        2. Geo880 grounded QDMR json files for the training set
    """
    assert data_split in ["train", "dev", "test"]
    prefix = f"GEO_{data_split}"
    # Read examples
    data = load_json(input_data)
    raw_data = data["data"]
    filtered_data = {"data": []}
    for example in raw_data:
        if example["example_id"].startswith(prefix):
            filtered_data["data"] += [example]
    num_left = len(filtered_data["data"])
    write_to_json(filtered_data, output_file)
    print(f"Done writing {num_left} examples to {output_file}.")
    return True


#geo_gold_sql_data = "../data/qdmr_data/groundings_geo880.json"
#geo_encoded_qdmr_data = "../data/qdmr_data/qdmr_ground_enc_geo880.json"
#geo_data_splits(geo_encoded_qdmr_data, "qdmr_ground_enc_geo880_train.json", "train")
#geo_data_splits(geo_gold_sql_data, "geo880_sql_train.json", "train")
#geo_data_splits(geo_gold_sql_data, "geo880_sql_dev.json", "dev")
#geo_data_splits(geo_gold_sql_data, "geo880_sql_test.json", "test")
