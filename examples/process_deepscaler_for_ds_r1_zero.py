from datasets import load_from_disk, Dataset
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.input) as f:
    samples = json.load(f)

ds = Dataset.from_list(samples)

print(f"loaded {len(ds)} samples")

def process_zero_2(sample):
    problem = sample["problem"]
    prompt = f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> tags, i.e., <think> reasoning process here </think> answer here. User: {problem} Please put your final answer within \\boxed{{}}. Assistant: "
    return {
        "prompt": prompt,
        "ground_truth": sample["answer"],
        "dataset_type": "deepscaler"
    }

processed_prompt = ds.map(process_zero_2)

processed_prompt.save_to_disk(args.output)