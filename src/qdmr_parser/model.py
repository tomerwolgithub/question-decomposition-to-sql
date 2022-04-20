import argparse
import os
import json
import logging
import re
from string import punctuation

import nltk

from dataset_qdmr import BreakDataset
from eval_qdmr.eval_string_match import StringMatch

nltk.download('punkt')

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup
)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        pl_logs = {"avg_train_loss": avg_train_loss}
        self.log_dict(pl_logs)
        return {"avg_train_loss": avg_train_loss}

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
                                       args=self.hparams,
                                       task=self.hparams.task,
                                       prepend_dataset_name=self.hparams.prepend_dataset_name)
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {**{"val_loss": avg_loss}, **results}
        pl_logs = {"epoch": self.trainer.current_epoch, "avg_val_loss": avg_loss,
                   "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
        pl_logs = {**pl_logs, **results}
        self.log_dict(pl_logs)
        return {"avg_val_loss": avg_loss}


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
        optimizer = Adafactor(
            model.parameters(),
            lr=self.hparams.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
        self.opt = optimizer
        return [optimizer]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
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
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


####### Dataset #######

def get_dataset(tokenizer, type_path, args):
    assert args.task in ["nl_to_qdmr"]
    data_file = args.training_set_file if type_path == "train" else args.dev_set_file
    return BreakDataset(tokenizer=tokenizer,
                        data_file=os.path.join(args.data_dir, data_file),
                        source_max_token_len=args.max_seq_length,
                        target_max_token_len=args.max_seq_length,
                        prepend_dataset_name=args.prepend_dataset_name)


####### Evaluation #######

def evaluate_predictions(examples, gold_labels, predictions, args, is_test=None, task=None,
                         prepend_dataset_name=None):
    evaluator = StringMatch()
    return evaluator.evaluate(questions=examples,
                              gold=gold_labels,
                              predict=predictions,
                              prepend_dataset_name=prepend_dataset_name)


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
                    if key not in ["progress_bar"]:
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
