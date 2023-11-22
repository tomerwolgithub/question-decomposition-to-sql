import csv
import json


def normalize_whitespace(source):
    tokens = source.split()
    return " ".join(tokens)


def load_json(filepath):
    with open(filepath, "r") as reader:
        text = reader.read()
    return json.loads(text)


def read_csv_to_dictionaries(filepath):
    with open(filepath, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        csv_examples = []
        for row in csv_reader:
            if line_count >= 0:
                csv_examples += [row]
            line_count += 1
        return csv_examples
