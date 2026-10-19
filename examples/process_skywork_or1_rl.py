from datasets import load_from_disk, Dataset
import argparse
import json
import re
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--difficulty-model", default="DeepSeek-R1-Distill-Qwen-32B")
parser.add_argument("--process-type", choices=["r1_zero", "sft"], default="sft")
parser.add_argument("--min-level", type=int, default=1)
parser.add_argument("--max-level", type=int, default=15)
parser.add_argument("--tokenizer", default=None)
args = parser.parse_args()

print(str(args))

samples = []
with open(args.input) as f:
    for raw_line in f:
        sample = json.loads(raw_line)
        difficulty_level = sample['extra_info']['model_difficulty'][args.difficulty_model]
        if difficulty_level >= args.min_level and difficulty_level <= args.max_level:
            samples.append(sample)

print(f"loaded {len(samples)} samples")

ds = Dataset.from_list(samples)

tokenizer = None
if args.tokenizer is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

def remove_question_number(text):
    match = re.match(r'^\d+\.\s*(.*)$', text)
    if match:
        return match.group(1)
    else:
        return text
    
def get_groud_truth(sample):
    try:
        gt = json.loads(sample['reward_model']['ground_truth'])
    except json.decoder.JSONDecodeError:
        gt = sample['reward_model']['ground_truth']
    # print(gt)
    if type(gt) == list:
        return str(gt[0])
    else:
        return str(gt) 

def process_sft(sample):
    global tokenizer
    problem = remove_question_number(sample["problem"])
    if tokenizer is not None:
        messages = [{
            "role": "user",
            "content": problem
        }]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
    return {
        "prompt": prompt,
        "ground_truth": get_groud_truth(sample),
        "dataset_type": "skywork_or1_rl"
    }

def process_r1_zero(sample):
    problem = remove_question_number(sample["problem"])
    prompt = f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> tags, i.e., <think> reasoning process here </think> answer here. User: {problem} Please put your final answer within \\boxed{{}}. Assistant: "
    return {
        "prompt": prompt,
        "ground_truth": get_groud_truth(sample),
        "dataset_type": "skywork_or1_rl"
    }

process = process_sft if args.process_type == 'sft' else process_r1_zero
processed_ds = ds.map(process, remove_columns=ds.column_names)

print(processed_ds[0])

processed_ds.save_to_disk(args.output)