import logging
logging.basicConfig(level=logging.INFO)

import datasets, copy
from transformers import AutoTokenizer
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", default=None)
parser.add_argument("--output-path", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--ignore-index", type=int, default=-100)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--max-length", type=int, default=8192)
parser.add_argument("--padding", action="store_true")
parser.add_argument("--truncation", action="store_true")
parser.set_defaults(padding=False, truncation=False, shift_labels=True)
args = parser.parse_args()

logging.info(args)

IGNORE_INDEX = args.ignore_index
logging.warning(f"IGNORE_INDEX: {IGNORE_INDEX}")

logging.warning(f"shift labels: {args.shift_labels}")

MAX_LENGTH = args.max_length
if args.shift_labels:
    MAX_LENGTH += 1
logging.warning(f"MAX_LENGTH: {MAX_LENGTH}")

logging.info(f"Loading data from {args.input_path}")

ds = datasets.load_dataset("json", data_files=args.input_path, split="train")

logging.info(f"{len(ds)} samples loaded")
logging.info(f"The last sample: \n{ds[-1]}")

logging.info(f"Loading tokenizer from {args.tokenizer}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
logging.info(f"Loaded tokenizer: \n{tokenizer}")

logging.warning(f"PAD_TOKEN_ID: {tokenizer.pad_token_id}")

def process_fn(sample, MAX_LENGTH=MAX_LENGTH, padding=args.padding, cut_off=args.truncation, shift_labels=args.shift_labels):
    input_ids = []
    labels = []

    # process prompt
    messages = [
        {"role": "user", "content": sample["query"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids.extend(prompt_ids)
    labels.extend([IGNORE_INDEX] * len(prompt_ids))

    # process answer
    answer_ids = tokenizer.encode(sample["response"], add_special_tokens=False)
    input_ids.extend(answer_ids)
    labels.extend(copy.deepcopy(answer_ids))

    if cut_off:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    # Padding
    pad_len = 0 if MAX_LENGTH is None else MAX_LENGTH - len(input_ids)
    if padding and pad_len > 0:
        # here pad_token_id should be different from pad token id
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len

    if shift_labels:
        input_ids = input_ids[:-1]
        labels = labels[1:]
    real_len = len(input_ids)
    res = {
        "input_ids": input_ids,
        "labels": labels,
        "real_len": real_len
    }
    return res

logging.info("Processing dataset")
tokenized_dataset = ds.map(process_fn, num_proc=args.num_workers, remove_columns=ds.column_names)

logging.info(f"Saving to {args.output_path}")
tokenized_dataset.save_to_disk(args.output_path)

logging.info(f"Last decode sample: \n{tokenizer.decode(tokenized_dataset[-1]['input_ids'])}")

logging.info("Content included in the loss:")
logging.info(tokenizer.decode(list(filter(lambda x: x!=-100, tokenized_dataset[-1]["labels"]))))

logging.info("done")
    