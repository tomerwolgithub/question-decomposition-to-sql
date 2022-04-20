import json
import collections

import torch
from torch.utils.data import Dataset, DataLoader
from utils_data import load_json, normalize_whitespace, read_csv_to_dictionaries

DATASET_DELIMITER = ":@@@"


def remove_dataset_delimiter_from_source(source):
    return source.split(DATASET_DELIMITER)[1].strip()


class BreakDataset(Dataset):
    def __init__(self, tokenizer, data_file, source_max_token_len=512, target_max_token_len=512,
                 prepend_dataset_name=None):
        self.data_file = data_file
        self.prepend_dataset_name = prepend_dataset_name

        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
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
        self._build_input_output(self.data_file)

    def _build_input_output(self, data_file):
        # Read examples
        raw_data = read_csv_to_dictionaries(data_file)
        for example in raw_data:
            # e.g., SPIDER_train_999 --> spider
            dataset = example["question_id"].split("_")[0].lower()
            source = example["question_text"]
            target = example["decomposition"]
            if self.prepend_dataset_name:
                source = "%s %s %s" % (dataset, DATASET_DELIMITER, source)
            target = normalize_whitespace(target)
            source += self.tokenizer.eos_token
            target += self.tokenizer.eos_token
            input = source.lower()
            target = target.lower()
            print("**** dataset input: ", input)
            print("**** dataset target: ", target) 

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.source_max_token_len,
                padding='max_length',
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target],
                max_length=self.target_max_token_len,
                padding='max_length',
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
