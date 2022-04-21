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

from model import LoggingCallback, T5FineTuner
import numpy as np
import torch
import pytorch_lightning as pl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args_dict = dict(
        dataset='geo',
        target_encoding='qdmr_sql',
        data_dir='data/spider',  # path for data files
        output_dir='',  # path to save the checkpoints
        db_dir='database',  # name of db dir in data dir
        training_set_file='qdmr_ground_enc_geo880_train.json',  # name of training set file in data dir
        dev_set_file='geo_dev.json',  # name of dev set file in data dir
        dev_set_sql='geo_dev.sql',  # name of file containing the dev sql queries in data dir
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

args_dict.update({'output_dir': 't5_geo_qdmr_formula_bs_2_accum_64_epochs_150_lr_1e-4',
                  'num_train_epochs': 150})
args = argparse.Namespace(**args_dict)

# logger = logging.getLogger(__name__)

# set random seed
set_seed(42)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    # filepath=args.output_dir, prefix="checkpoint", monitor="train_loss", mode="min", save_top_k=1
    filepath=args.output_dir, prefix="checkpoint", monitor="exec_acc", mode="max", save_top_k=1
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

# initialize model
model = T5FineTuner(args)

# initialize trainer
trainer = pl.Trainer(**train_params)

# start fine-tuning
trainer.fit(model)