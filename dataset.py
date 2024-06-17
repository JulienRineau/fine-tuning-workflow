import json
import logging
import os
import random

import pandas as pd
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb

with open("config.json", "r") as file:
    config = json.load(file)
    token = config["hf_token"]

login(token=token)

with open("alpaca_gpt4_data.json", "r") as f:
    alpaca = json.load(f)


def save_jsonl(data, filename):
    with open(filename, "w") as file:
        for entry in data:
            json.dump(entry, file)
            file.write("\n")


def prompt_no_input(row):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ).format_map(row)


def prompt_input(row):
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ).format_map(row)


def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)


def pack(tokenizer, dataset, max_seq_len=1024):
    tkds_ids = tokenizer([s["example"] for s in dataset])["input_ids"]

    all_token_ids = []
    for tokenized_input in tkds_ids:
        all_token_ids.extend(tokenized_input + [tokenizer.eos_token_id])

    packed_ds = []
    for i in range(0, len(all_token_ids), max_seq_len + 1):
        input_ids = all_token_ids[i : i + max_seq_len + 1]
        if len(input_ids) == (max_seq_len + 1):
            packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})
    return packed_ds


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    wandb_project = "qwen-fine-tuning"
    model_id = "Qwen/Qwen2-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    run = wandb.init(project=wandb_project, job_type="full_process")

    with open("alpaca_gpt4_data.json", "r") as f:
        alpaca = json.load(f)
        logging.info(f"Loaded alpaca dataset with {len(alpaca)} entries")

    prompts = [create_prompt(row) for row in tqdm(alpaca, desc="Creating prompts")]
    EOS_TOKEN = "</s>"
    outputs = [row["output"] + EOS_TOKEN for row in alpaca]

    dataset = [
        {"prompt": s, "output": t, "example": s + t} for s, t in zip(prompts, outputs)
    ]
    random.shuffle(dataset)
    logging.info("Dataset shuffled")
    train_dataset = dataset[:-1000]
    eval_dataset = dataset[-1000:]
    logging.info(
        f"Dataset split into {len(train_dataset)} training and {len(eval_dataset)} evaluation samples"
    )

    with wandb.init(
        project=wandb_project, job_type="create_dataset", reinit=True
    ) as dataset_run:
        at = wandb.Artifact(
            name="alpaca_gpt4",
            type="dataset",
            description="A GPT4 generated Alpaca-like dataset for instruction finetuning",
            metadata={
                "url": "https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#how-good-is-the-data"
            },
        )
        at.add_file("alpaca_gpt4_data.json")
        wandb.log_artifact(at)

        table = wandb.Table(columns=list(alpaca[0].keys()))
        for row in tqdm(alpaca, desc="Logging data to WandB"):
            table.add_data(*row.values())
        wandb.log({"complete_dataset": table})

    train_ds_packed = pack(tokenizer, train_dataset)
    eval_ds_packed = pack(tokenizer, eval_dataset)

    save_jsonl(train_ds_packed, "train_packed_alpaca.jsonl")
    save_jsonl(eval_ds_packed, "eval_packed_alpaca.jsonl")

    with wandb.init(
        project=wandb_project, job_type="split_data", reinit=True
    ) as split_run:
        train_table = wandb.Table(dataframe=pd.DataFrame(train_dataset))
        eval_table = wandb.Table(dataframe=pd.DataFrame(eval_dataset))
        wandb.log({"train_dataset": train_table, "eval_dataset": eval_table})

    with wandb.init(
        project=wandb_project, job_type="preprocess", reinit=True
    ) as preprocess_run:
        packed_at = wandb.Artifact(
            name="packed_alpaca",
            type="dataset",
            description="Alpaca dataset packed in sequences",
            metadata={"max_seq_len": 1024, "model_id": model_id},
        )
        packed_at.add_file("train_packed_alpaca.jsonl")
        packed_at.add_file("eval_packed_alpaca.jsonl")
        wandb.log_artifact(packed_at)

    logging.info("WandB logging complete")
    run.finish()
