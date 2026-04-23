from datasets import load_from_disk, Dataset
import argparse
import json
import re
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--tokenizer", required=True)
args = parser.parse_args()

print(str(args))

with open(args.input) as f:
    samples = json.load(f)

ds = Dataset.from_list(samples)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

def process_fn(sample):
    global tokenizer
    problem = sample["problem"]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": problem
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return {
        "prompt": prompt,
        "ground_truth": sample["answer"],
        "dataset_type": "ignore"
    }

processed_ds = ds.map(process_fn, remove_columns=ds.column_names)

print(processed_ds[0])

processed_ds.save_to_disk(args.output)