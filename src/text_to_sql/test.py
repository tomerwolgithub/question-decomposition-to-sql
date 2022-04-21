import textwrap
from tqdm.auto import tqdm
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk

from dataset_utils import load_json

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from model import LoggingCallback, T5FineTuner, get_dataset, evaluate_predictions
import numpy as np
import torch
import pytorch_lightning as pl

if __name__ == '__main__':
    args_dict = dict(
        dataset='geo',
        target_encoding='qdmr_formula',
        data_dir='data/spider',  # path for data files
        output_dir='',  # path to save the checkpoints
        db_dir='other_database',  # name of db dir in data dir
        training_set_file='train_spider.json',  # name of training set file in data dir
        dev_set_file='other_queries/yelp/yelp_sql.json',  # name of dev set file in data dir
        dev_set_sql='other_queries/yelp/test_gold_yelp.sql',  # name of file containing the dev sql queries in data dir
        test_set_file='',
        test_set_sql='',
        model_name_or_path='t5-base',
        tokenizer_name_or_path='t5-base',
        max_seq_length=512,
        learning_rate=1e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=2,
        eval_batch_size=2,
        num_train_epochs=2,
        gradient_accumulation_steps=64,
        n_gpu=1,
        early_stop_callback=False,
        fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

args = argparse.Namespace(**args_dict)

tokenizer = T5Tokenizer.from_pretrained('t5-large')
checkpoint_path = "/t5_large_spider-train_qdmr_formula_few_shot_finetune_135_epochs_bs_1_accum_128_epochs_150_lr_1e-4_seed_42/checkpointepoch=9.ckpt"

print("Loading T5FinalTuner pretrained model...")
model = T5FineTuner.load_from_checkpoint(checkpoint_path)

# re-set missing params
model.hparams.target_encoding = args.target_encoding
model.hparams.data_dir = args.data_dir
model.hparams.db_dir = args.db_dir
model.hparams.dev_set_file = args.dev_set_file
model.hparams.dev_set_sql = args.dev_set_sql

model = model.to('cuda')
print("Done!")

print("Loading dataset...")
# dataset = get_dataset(tokenizer, "test", args)
dataset = get_dataset(tokenizer=tokenizer,
                      type_path="val",
                      args=args,
                      dataset_name=args.dataset,
                      target_encoding=args.target_encoding)

print("Generating sequences...")
loader = DataLoader(dataset, batch_size=1, num_workers=4)
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
    outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                attention_mask=batch['source_mask'].cuda(),
                                max_length=512)

    dec = [tokenizer.decode(ids) for ids in outs]
    target = [tokenizer.decode(ids) for ids in batch["target_ids"]]

    outputs.extend(dec)
    targets.extend(target)


def restore_oov(prediction):
    """
    Replace T5 SPM OOV character with `<`.
    Certain punctuation characters are mapped to the OOV symbol in T5's
    sentence-piece model. For Spider, this appears to only affect the `<` symbol,
    so it can be deterministically recovered by running this script.
    An alternative is to preprocess dataset to avoid OOV symbols for T5.
    """
    pred = prediction.replace(" Ã¢Ââ€¡ ", "<")
    return pred


def format_sql(prediction):
    pred = restore_oov(prediction)
    return pred.split(";")[0] + ";"


print(len(outputs))

for i, out in enumerate(outputs):
    print(out)

#print("****************** Formatted *************************")
#for i, out in enumerate(outputs):
#    print(format_sql(out))


dbs_list = ["geo:"] * len(outputs)
if args.dataset == "spider":
    dbs_list = []
    dev_file_path = os.path.join(args.data_dir, args.dev_set_file)
    raw_data = load_json(dev_file_path)
    for example in raw_data:
        dbs_list += [example["db_id"]]

score = evaluate_predictions(examples=dbs_list,
                             gold_labels=args.test_set_sql,
                             predictions=outputs,
                             args=args,
                             is_test=True)
print(score)