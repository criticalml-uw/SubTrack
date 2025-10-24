# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import time

import datasets
import evaluate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForSequenceClassification
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from low_rank_torch import LowRankAdamW

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.38.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    'boolq': ("question", "passage", None, None),  # TODO: Is this order correct?
    'cb': ("premise", "hypothesis", None, None),
    'copa': ("premise", "question", "choice1", "choice2"),
    'multirc': ("paragraph", "question", "answer", None),
    # 'record': (), # TODO: figure it out
    'rte': ("premise", "hypothesis", None, None),
    'wic': ("sentence1", "sentence2", "word", None),
    'wsc': ("text", "span1_text", "span2_text", None),
    # 'wsc.fixed': ("text", "span1_text", "span2_text"),
    'axb': ("sentence1", "sentence2", None, None),
    'axg': ("premise", "hypothesis", None, None)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--load_pretrained_model", type=str, default=None)

    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg'],
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--enable_warmup", action="store_true", help="Whether or not to use warmup.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument("--enable_low_rank", action="store_true", help="Whether or not to use low rank optimizer.")
    parser.add_argument("--subspace_update_interval", type=int, default=50)
    parser.add_argument("--low_rank_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument("--subspace_update_method", type=str, choices=["galore", "subtrack"])
    parser.add_argument("--adaptive_optimizer", action="store_true",
                        help="Whether to use adaptive optimizer for low-rank optimizers")
    parser.add_argument("--recovery_scaling", action="store_true")
    parser.add_argument("--norm_growth_limiter_off", action="store_true")
    parser.add_argument("--norm_growth_limit",  type=float, default=1.01)
    ##################################################################################################################
    # Subspace Tracking Parameters
    parser.add_argument("--st_init_step_size", type=float, default=1e-3)
    parser.add_argument(
        "--st_step_size_scheduler", type=str, default="constant",
        choices=["constant", "iterative_decrease", "adaptive"]
    )
    parser.add_argument("--st_step_size_coef", type=float, default=1.0)
    parser.add_argument("--st_noise_sigma2", type=float, default=0)
    parser.add_argument("--st_subspace_coef", type=float, default=1.0)
    ##################################################################################################################
    # lora_all_modules
    parser.add_argument("--lora_all_modules", action="store_true", help="Whether or not to use lora for all modules.")
    # eval_llama
    parser.add_argument("--eval_llama", action="store_true", help="Whether or not to evaluate llama model.")
    # low_rank_method
    parser.add_argument("--low_rank_method", type=str, default=None, help="low rank method for wandb sweep")

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main(sweep_config=None):
    send_example_telemetry("run_glue_no_trainer", args)
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.enable_low_rank:
        wandb.init(
            project="SubTrack++SUPER_GLUE",
            name=f"{args.task_name}-{args.subspace_update_method}"
        )
    else:
        wandb.init(
            project="SubTrack++SUPER_GLUE",
            name=f"{args.task_name}-full-rank"
        )
    wandb.config.update(dict(vars(args)), allow_val_change=True)

    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.task_name is not None:
        raw_datasets = load_dataset("aps/super_glue", args.task_name, trust_remote_code=True)
        if args.task_name == 'axb' or args.task_name == 'axg':
            raw_datasets = raw_datasets['test'].train_test_split(test_size=0.3)
            test_dataset = raw_datasets['test'].train_test_split(test_size=0.5)
            raw_datasets['test'] = test_dataset['train']
            raw_datasets['validation'] = test_dataset['test']
            del test_dataset
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    if not args.eval_llama:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
        if args.task_name == 'copa':
            model = AutoModelForMultipleChoice.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                ignore_mismatched_sizes=args.ignore_mismatched_sizes,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                ignore_mismatched_sizes=args.ignore_mismatched_sizes,
                trust_remote_code=args.trust_remote_code,
            )
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        setattr(config, 'num_labels', num_labels)
        setattr(config, 'finetuning_task', args.task_name)
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
        tokenizer.padding_side = "left"
        model = LlamaForSequenceClassification(
            config
        )

    if args.load_pretrained_model:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.load_pretrained_model}")
        checkpoint_path = os.path.join(args.load_pretrained_model, "pytorch_model.bin")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for key in checkpoint.keys():
            if key not in model.state_dict().keys():
                print(f"key {key} not in model state dict")

        for key in model.state_dict().keys():
            if key not in checkpoint.keys():
                print(f"key {key} not in checkpoint")
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Model successfully loaded (strict=False policy)")
        logger.info("*" * 40)

    if not args.lora_all_modules:
        target_modules_list = ["q_proj", "v_proj"]
    else:
        print('Enabling LoRA for all modules')
        target_modules_list = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "k_proj", "o_proj"]

    if 'bert' in args.model_name_or_path:
        if not args.lora_all_modules:
            target_modules_list = ["query"]
        else:
            print('Enabling LoRA for all modules')
            target_modules_list = ["query", "value", "key", "intermediate.dense", "output.dense"]

    if args.task_name is not None:
        sentence1_key, sentence2_key, sentence3_key, sentence4_key = task_to_keys[args.task_name]
    else:  # TODO: it is the same as GLUE, we should adopt it to SuperGLUE
        raise "You Shouldn't be here!"
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and args.task_name is not None
            and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        if args.task_name == 'wsc':
            process_wsc = lambda words, ind1, word1, ind2, word2: ' '.join(
                words[:ind1] + ['<s>' + word1 + '</s>'] + words[ind1 + len(word1.split(' ')):ind2] + [
                    '<s>' + word2 + '</s>'] + words[ind2 + len(word2.split(' ')):])
            texts = (
                ([process_wsc(text.split(' '), idx1, word1, idx2, word2) for text, idx1, word1, idx2, word2 in
                  zip(examples['text'], examples['span1_index'], examples['span1_text'], examples['span2_index'],
                      examples['span2_text'])],)
            )
        elif args.task_name == 'copa':
            premises = [[premise]*2 for premise in examples['premise']]
            alternatives = [[choice1, choice2] for choice1, choice2 in zip(examples['choice1'], examples['choice2'])]
            premises = sum(premises, [])
            alternatives = sum(alternatives, [])
            result = tokenizer(premises, alternatives, padding=True, max_length=args.max_length, truncation=True)
            result = {k: [v[i: i + 2] for i in range(0, len(v), 2)] for k, v in result.items()}
        elif sentence3_key is None:
            texts = (
                (examples[sentence1_key], examples[sentence2_key],)
            )
        elif sentence4_key is None:
            texts = (
                (examples[sentence1_key], examples[sentence2_key], examples[sentence3_key],)
            )
        else:
            texts = (
                (examples[sentence1_key], examples[sentence2_key], examples[sentence3_key], examples[sentence4_key])
            )
        if args.task_name != 'copa':
            result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if Accelerator.mixed_precision == 'fp16' else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if not args.enable_low_rank:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        from torch import nn
        low_rank_params = []
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable low rank for weights in module: ', module_name)
            low_rank_params.append(module.weight)

        id_low_rank_params = [id(p) for p in low_rank_params]
        regular_params = [p for p in model.parameters() if id(p) not in id_low_rank_params]
        param_groups = [{'params': regular_params},
                        {'params': low_rank_params, 'rank': args.lora_r,
                         'scale': args.low_rank_scale, 'proj_type': args.proj_type,
                         'subspace_update_method': args.subspace_update_method,
                         'st_init_step_size': args.st_init_step_size,
                         'st_step_size_scheduler': args.st_step_size_scheduler,
                         'st_step_size_coef': args.st_step_size_coef,
                         'st_noise_sigma2': args.st_noise_sigma2,
                         'st_subspace_coef': args.st_subspace_coef,
                         'subspace_update_interval': args.subspace_update_interval,
                         'adaptive_optimizer': args.adaptive_optimizer,
                         'recovery_scaling': args.recovery_scaling,
                         'norm_growth_limit': args.norm_growth_limit,
                         'norm_growth_limiter_off': args.norm_growth_limiter_off,}]
        optimizer = LowRankAdamW(param_groups, lr=args.learning_rate, betas=(0.908, 0.99), eps=1e-8)

        n_params = sum(p.numel() for p in model.parameters())
        n_low_rank_params = sum(p.numel() for p in low_rank_params)
        wandb.config.update({'n_params': n_params, 'n_low_rank_params': n_low_rank_params})

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.enable_warmup:
        args.num_warmup_steps = args.num_warmup_steps if args.num_warmup_steps > 0 else 0.1*args.max_train_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    wandb.config.update({
        'lr_scheduler_type': args.lr_scheduler_type,
        'num_warmup_steps': args.num_warmup_steps,
        'num_training_steps': args.max_train_steps
    }, allow_val_change=True)

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    if args.task_name is not None:
        metric = evaluate.load("super_glue", args.task_name, trust_remote_code=True)
    else:
        metric = evaluate.load("accuracy")

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)

    t_start = time.time()
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        eval_raw_dataloader = DataLoader(raw_datasets['validation'], batch_size=args.per_device_eval_batch_size)
        for step, (batch, raw_batch) in enumerate(zip(eval_dataloader, eval_raw_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]

            if args.task_name == 'multirc':
                metric.add_batch(
                    predictions=[{'idx': {
                        'question': raw_batch['idx']['question'][i],
                        'paragraph': raw_batch['idx']['paragraph'][i],
                        'answer': raw_batch['idx']['answer'][i]
                        }, 'prediction': p} for i, p in enumerate(predictions)],
                    references=references,
                )
            else:
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        wandb.log({
            "performance": eval_metric,
            "train_loss": total_loss.item() / len(train_dataloader),
            "epoch": epoch,
            "step": completed_steps,
        })

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    print()
    print('=' * 100)
    print(f'Training and between validation step took {time.time() - t_start}')
    print('=' * 100)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    args = parse_args()
    main()
