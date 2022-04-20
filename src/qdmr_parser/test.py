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
import sys
from itertools import chain
from string import punctuation

import nltk

from eval_qdmr.eval_string_match import format_prediction

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
        predictions_output_file='t5_large_qdmr_parser_spider_train.txt',
        checkpoint_path='/trained_models/t5_large_qdmr_parser_bs_2_accum_64_epochs_150_lr_1e-4/epoch=6-exact_match=0.231.ckpt',
        task='nl_to_qdmr',
        data_dir='',  # path for data files
        output_dir='',  # path to save the checkpoints
        training_set_file='train.csv',  # name of training set file in data dir
        dev_set_file='dev.csv',  # name of dev set file in data dir
        dev_set_labels='',  # name of file containing the dev qdmr/nl queries in data dir
        test_set_file='',
        test_set_labels='',
        model_name_or_path='t5-large',
        tokenizer_name_or_path='t5-large',
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
        fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args = argparse.Namespace(**args_dict)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    print("Loading T5FinalTuner pretrained model...")
    model = T5FineTuner.load_from_checkpoint(args.checkpoint_path)
    model = model.to('cuda')
    print("Done!")

    # re-set dataset params
    model.hparams.dev_set_file = args.dev_set_file

    print("Loading dataset...")
    dataset = get_dataset(tokenizer=tokenizer,
                          type_path="test",
                          args=model.hparams)

    print("Generating sequences...")
    loader = DataLoader(dataset, batch_size=model.hparams.eval_batch_size, num_workers=2)
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

    with open(args.predictions_output_file, 'w') as f:
        for i, out in enumerate(outputs):
            formatted_out = format_prediction(out)
            print(formatted_out, file=f)
