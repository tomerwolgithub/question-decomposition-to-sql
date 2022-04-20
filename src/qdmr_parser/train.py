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
        task='nl_to_qdmr',
        data_dir='',  # path for data files
        output_dir='',  # path to save the checkpoints
        training_set_file='',  # name of training set file in data dir
        dev_set_file='',  # name of dev set file in data dir
        dev_set_labels='',  # name of file containing the dev qdmr/nl queries in data dir
        test_set_file='',
        test_set_labels='',
        prepend_dataset_name=True,
        model_name_or_path='t5-large',
        tokenizer_name_or_path='t5-large',
        max_seq_length=512,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=0,
        train_batch_size=2,  
        eval_batch_size=2, 
        num_train_epochs=15,
        gradient_accumulation_steps=64, 
        n_gpu=1,
        fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

args_dict.update({'data_dir': 'data/break',
                  'training_set_file': 'train.csv',
                  'dev_set_file': 'dev.csv',
                  'output_dir': 't5_large_qdmr_parser_bs_2_accum_64_epochs_10_lr_1e-4'})

args = argparse.Namespace(**args_dict)

# set random seed
set_seed(42)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, filename='{epoch}-{exact_match:.3f}',
    monitor="exact_match", mode="max", save_top_k=1, save_last=False
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=True,
    callbacks=[LoggingCallback(), checkpoint_callback],
)

# initialize model
model = T5FineTuner(args)

# Load pre-trained model from checkpoint
# checkpoint_path = "/trained_models/t5_large_qdmr_parser_bs_2_accum_64_epochs_150_lr_1e-4/last.ckpt"
# print("Loading T5FinalTuner pretrained model...")
# model = T5FineTuner.load_from_checkpoint(checkpoint_path)
# for key in args_dict.keys():
#     model.hparams[key] = args_dict[key]
# model = model.to('cuda')
# print("Done!")

# initialize trainer
trainer = pl.Trainer(**train_params)

# start fine-tuning
trainer.fit(model)