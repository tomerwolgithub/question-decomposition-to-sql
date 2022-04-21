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

from eval_spider import ExactSetMatch, ExecutionAccuracy
from dataset_spider import SpiderDataset
from dataset_qdmr import QDMRDataset


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

        self.target_encoding = hparams.target_encoding
        self.dataset_name = hparams.dataset

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # record the source, targets & predictions for each batch
        # use these to compute evaluate on the validation set
        src = [self.tokenizer.decode(ids) for ids in batch['source_ids']]
        target = [self.tokenizer.decode(ids) for ids in batch['target_ids']]
        preds = self.model.generate(input_ids=batch['source_ids'].cuda(),
                                    attention_mask=batch['source_mask'].cuda(),
                                    max_length=512)
        dec_preds = [self.tokenizer.decode(ids) for ids in preds]
        loss = self._step(batch)
        return {"val_loss": loss,
                "source": src,
                "target": target,
                "preds": dec_preds}

    def validation_epoch_end(self, outputs):
        # evaluate predictions of the entire validation set
        all_inputs = []
        all_targets = []
        all_predictions = []
        for val_step_out in outputs:
            all_inputs.extend(val_step_out["source"])
            all_targets.extend(val_step_out["target"])
            all_predictions.extend(val_step_out["preds"])
        results = evaluate_predictions(examples=all_inputs,
                                       gold_labels=all_targets,
                                       predictions=all_predictions,
                                       args=self.hparams)
        ex_score = results["exec"]
        errors = results["errors"]
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "exec_acc": ex_score, "errors": errors}
        return {"exec_acc": ex_score, "errors": errors, "avg_val_loss": avg_loss,
                "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def dummy_val_epoch_eval(self, inputs, targets, predictions):
        score = 0
        for i in range(len(targets)):
            if targets[i] == predictions[i]:
                score += 1
        return float(score) / len(targets)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer,
                                    type_path="train",
                                    args=self.hparams,
                                    dataset_name=self.dataset_name,
                                    target_encoding=self.target_encoding)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer,
                                  type_path="val",
                                  args=self.hparams,
                                  dataset_name=self.dataset_name,
                                  target_encoding=self.target_encoding)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


####### Dataset #######

def get_dataset(tokenizer, type_path, args, dataset_name, target_encoding):
    if dataset_name == "spider":
        if target_encoding in [None, "sql"]:
            return get_spider_dataset(tokenizer, type_path, args)
        return get_qdmr_dataset(tokenizer, type_path, args, target_encoding)
    if dataset_name == "geo":
        return get_geo_dataset(tokenizer, type_path, args, target_encoding)
    return None


def get_spider_dataset(tokenizer, type_path, args):
    data_json = args.training_set_file if type_path == "train" else args.dev_set_file
    return SpiderDataset(tokenizer=tokenizer,
                         data_file=os.path.join(args.data_dir, data_json),
                         tables_file=os.path.join(args.data_dir, "tables.json"),
                         dataset_type=type_path,
                         max_len=args.max_seq_length,
                         append_schema=True)


def get_qdmr_dataset(tokenizer, type_path, args, target_encoding):
    if type_path in ["test", "val"]:
        data_split = args.dev_set_file if type_path == "val" else args.test_set_file
        return SpiderDataset(tokenizer=tokenizer,
                             data_file=os.path.join(args.data_dir, data_split),
                             tables_file=os.path.join(args.data_dir, "tables.json"),
                             dataset_type=type_path,
                             max_len=args.max_seq_length,
                             append_schema=True)
    return QDMRDataset(tokenizer=tokenizer,
                       data_file=os.path.join(args.data_dir, args.training_set_file),
                       tables_file=os.path.join(args.data_dir, "tables.json"),
                       dataset_type=type_path,
                       max_len=args.max_seq_length,
                       append_schema=True,
                       encoding=target_encoding)


def get_geo_dataset(tokenizer, type_path, args, target_encoding):
    # evaluate on gold sql targets from the grounded qdmr file
    encoding = "sql" if type_path in ["test", "val"] else target_encoding
    data_json = args.training_set_file
    if type_path in ["test", "val"]:
        data_json = args.dev_set_file if type_path == "val" else args.test_set_file
    return QDMRDataset(tokenizer=tokenizer,
                       data_file=os.path.join(args.data_dir, data_json),
                       tables_file=os.path.join(args.data_dir, "tables.json"),
                       dataset_type=type_path,
                       max_len=args.max_seq_length,
                       append_schema=True,
                       encoding=encoding)


####### Evaluation #######

# ESM evaluator
# def evaluate_predictions(examples, gold_labels, predictions, args):
#     evaluator = ExactSetMatch()
#     gold_file = os.path.join(args.data_dir, "dev_gold.sql")
#     with open(gold_file) as f:
#         glist = [l.strip().split('\t')[0] for l in f.readlines() if len(l.strip()) > 0]
#     return evaluator.evaluate(examples=examples,
#                               gold=glist,
#                               predict=predictions,
#                               db_dir=os.path.join(args.data_dir, "database"),
#                               table=os.path.join(args.data_dir, "tables.json"),
#                               etype="match")

# Execution set match evaluator
def evaluate_predictions(examples, gold_labels, predictions, args, is_test=None):
    evaluator = ExecutionAccuracy()
    labels_file = args.test_set_sql if is_test else args.dev_set_sql
    gold_file = os.path.join(args.data_dir, labels_file)
    with open(gold_file) as f:
        glist = [l.strip().split('\t')[0] for l in f.readlines() if len(l.strip()) > 0]
    return evaluator.evaluate(examples=examples,
                              gold=glist,
                              predict=predictions,
                              db_dir=os.path.join(args.data_dir, args.db_dir),
                              exec_set_match=True,
                              pred_type=args.target_encoding,
                              dataset=args.dataset)


####### Logger #######

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "val_results.txt")
            with open(output_test_results_file, "a") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))
