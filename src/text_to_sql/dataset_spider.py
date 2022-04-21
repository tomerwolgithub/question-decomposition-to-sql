import json
import collections

import torch
from torch.utils.data import Dataset, DataLoader
from dataset_utils import load_json, normalize_whitespace


class SpiderDataset(Dataset):
    def __init__(self, tokenizer, data_file, tables_file, dataset_type,
                 max_len=512, append_schema=None):
        self.dataset_type = dataset_type
        self.data_file = data_file
        self.tables_file = tables_file
        self.append_schema = append_schema

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        self._build_input_output(self.data_file, self.tables_file, self.append_schema)

    def _build_input_output(self, data_file, tables_file, append_schema):
        if append_schema:
            # Serialize DB schema
            tables_json = load_json(tables_file)
            db_id_to_schema_string = {}
            for table_json in tables_json:
                db_id = table_json["db_id"].lower()
                db_id_to_schema_string[db_id] = self._get_schema_string(table_json)
        # Read examples
        raw_data = load_json(data_file)
        for example in raw_data:
            database = example["db_id"].lower()
            source = example["question"]
            target = example["query"]
            # Prepend database
            source = "%s: %s" % (database, source)
            if append_schema:
                schema_string = db_id_to_schema_string[database]
                source = "%s%s" % (source, schema_string)
            target = normalize_whitespace(target)
            source += self.tokenizer.eos_token
            target += self.tokenizer.eos_token
            input = source.lower()
            target = target.lower()

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

    def _get_schema_string(self, table_json):
        """Returns the schema serialized as a string."""
        table_id_to_column_names = collections.defaultdict(list)
        for table_id, name in table_json["column_names_original"]:
            table_id_to_column_names[table_id].append(name.lower())
        tables = table_json["table_names_original"]

        table_strings = []
        for table_id, table_name in enumerate(tables):
            column_names = table_id_to_column_names[table_id]
            table_string = " | %s : %s" % (table_name.lower(), " , ".join(column_names))
            table_strings.append(table_string)

        return "".join(table_strings)