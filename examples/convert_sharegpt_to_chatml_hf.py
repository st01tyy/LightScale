# 该脚本用于将一个目录下的所有sharegpt格式对话数据json文件切词并套用chatml对话模板，处理成huggingface datasets格式的数据集

import os
cache_dir = os.makedirs("./.cache", exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = "./.cache"
os.environ['HF_DATASETS_CACHE'] = "./.cache"

import datasets, copy
from transformers import AutoTokenizer
import numpy as np
import argparse
import logging
import re

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", default=None)
parser.add_argument("--output-path", required=True)
parser.add_argument("--tokenizer", default="/root/work/externalstorage/gpfsprd/JT_Pretrain_Models/75Bv2_tokenizer")
parser.add_argument("--field", default="conversations")
parser.add_argument("--ignore-index", type=int, default=-100)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--max-length", type=int, default=8192)
parser.add_argument("--padding", action="store_true")
parser.add_argument("--truncation", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--shift-labels", action="store_true")
parser.add_argument("--input-is-dir", action="store_true")
parser.add_argument("--mask-search-result", action="store_true")
parser.set_defaults(padding=False, truncation=False, debug=False, shift_labels=False, input_is_dir=False, mask_search_result=False)
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
if args.input_is_dir:
    ds = datasets.load_dataset("json", data_dir=args.input_path, split="train")
else:
    ds = datasets.load_dataset("json", data_files=args.input_path, split="train")
if args.field:
    ds = ds.select_columns(args.field)
if args.debug:
    ds = ds.select(range(0, 64))
logging.info(f"{len(ds)} samples loaded")
logging.info(f"The last sample: \n{ds[-1]}")

logging.info(f"Loading tokenizer from {args.tokenizer}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
logging.info(f"Loaded tokenizer: \n{tokenizer}")

logging.warning(f"PAD_TOKEN_ID: {tokenizer.pad_token_id}")

# def preprocess(example):
#     data = {"conversations": example["conversations"]}
#     return data

def _process_assistant_content_with_search_result(content: str):
    if '<begin_search>' in content or '<begin_click>' in content:
        label = []
        pattern = r'(<search_result>(.*?)</search_result>)|(<click_result>(.*?)</click_result>)'
        last_end_id = 0
        sentence_ids = []
        for m in re.finditer(pattern, content, re.DOTALL):
            # 为了格式解析，<search_result>和<code_result>计入loss
            start_id, end_id = m.start(2), m.end(2)
            if start_id == -1 and end_id == -1:
                start_id, end_id = m.start(4), m.end(4)
            cur_sentence_ids = tokenizer.encode(content[last_end_id:start_id])
            label += copy.deepcopy(cur_sentence_ids)
            sentence_ids += copy.deepcopy(cur_sentence_ids)
            # 由<search_result>和</search_result>、<code_result>和</code_result>包裹的部分不计入loss
            cur_sentence_ids = tokenizer.encode(content[start_id:end_id])
            label += [IGNORE_INDEX] * len(cur_sentence_ids)
            sentence_ids += copy.deepcopy(cur_sentence_ids)
            last_end_id = end_id
        if last_end_id < len(content) - 1:
            cur_sentence_ids = tokenizer.encode(content[last_end_id:])
            label += copy.deepcopy(cur_sentence_ids)
            sentence_ids += copy.deepcopy(cur_sentence_ids)
    else:
        sentence_ids = tokenizer.encode(content)
        label = copy.deepcopy(sentence_ids)
    return sentence_ids, label

"""
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt 1}<|im_end|>
<|im_start|>assistant
{reply 1}<|im_end|>
<|im_start|>user
{prompt 2}<|im_end|>
<|im_start|>assistant
(...)
"""
def generate_and_tokenize_prompt(data_point, field=args.field, MAX_LENGTH=MAX_LENGTH, padding=args.padding, cut_off=args.truncation, shift_labels=args.shift_labels):
    input_ids = []
    labels = []
    source = data_point[field]
    for idx, sentence in enumerate(source):
        sentence_from = sentence["from"].lower().strip()
        if sentence_from == "system":
            sentence_value = "<|im_start|>system\n" + sentence["value"] + "<|im_end|>"
        elif sentence_from == "human" or sentence_from == "user":
            sentence_value = "\n<|im_start|>user\n" + sentence["value"] + "<|im_end|>\n<|im_start|>assistant\n"
        elif sentence_from == "observation":
            sentence_value = "\n<|im_start|>observation\n" + sentence["value"] + "<|im_end|>\n<|im_start|>assistant\n"
        elif sentence_from == "assistant" or sentence_from == "masked_assistant":
            sentence_value = sentence["value"] + "<|im_end|>"
        else:
            raise RuntimeError(f"unexpected role: {sentence_from}")

        if sentence_from == "assistant" and args.mask_search_result:
            sentence_ids, label = _process_assistant_content_with_search_result(sentence_value)
        else:
            sentence_ids = tokenizer.encode(sentence_value)
            #print("idx : ", idx, ", sentence_value : ", repr(sentence_value)[1:-1],  ", sentence_ids : ", sentence_ids, sentence_from == "system")
            label = copy.deepcopy(sentence_ids) if sentence_from == "assistant" else [IGNORE_INDEX] * len(sentence_ids)

        input_ids += sentence_ids
        labels += label
    # real_len = len(input_ids)
    if cut_off:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    # real_len = len(input_ids)

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
    tokenized_full_prompt = {
        "input_ids": input_ids,
        "labels": labels,
        "real_len": real_len
    }
    return tokenized_full_prompt

logging.info("Processing dataset")
tokenized_dataset = ds.map(generate_and_tokenize_prompt, num_proc=args.num_workers, remove_columns=ds.column_names)

logging.info(f"Saving to {args.output_path}")
tokenized_dataset.save_to_disk(args.output_path)

logging.info(f"Last decode sample: \n{tokenizer.decode(tokenized_dataset[-1]['input_ids'])}")
if args.mask_search_result:
    logging.info('======== 计入loss的token：')
    logging.info(tokenizer.decode(list(filter(lambda x: x!=-100, tokenized_dataset[-1]["labels"]))))

logging.info("done")
