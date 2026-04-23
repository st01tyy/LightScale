import os

from datasets import load_from_disk, Dataset
import argparse
import logging
from typing import List
from tqdm import tqdm
import numpy as np
import os

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", required=True)
parser.add_argument("--process-type", choices=["shuffle", "sort"], default="shuffle")
parser.add_argument("--target-length-in-k", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--plus-one", action="store_true")
parser.add_argument("--pad-token-id", required=True)
parser.add_argument("--ignore-index", type=int, default=-100)
parser.add_argument("--packing-mode", type=int, choices=[0, 1, 2], default=0)
parser.add_argument("--batch-size", type=int, default=0)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--override-target-length", type=int, default=None)
parser.set_defaults(debug=False, plus_one=False)
args = parser.parse_args()

"""
packing_mode介绍
0: 只构造cu_seqlens，不构造cu_seqlens_padded；将padding部分视为正常序列的一部分
1：只构造cu_seqlens，不构造cu_seqlens_padded；将padding部分视为独立的的一个序列片段
2：既构造cu_seqlens，也构造cu_seqlens_padded；

举例：
A A B B B PAD PAD
s
mode 0:
cu_seqlens: [0, 2, 7]
cu_seqlens_padded: None

mode 1:
cu_seqlens: [0, 2, 5, 7]
cu_seqlens_padded: None

mode 2:
cu_seqlens: [0, 2, 5]
cu_seqlens_padded: [0, 2, 7]
"""

logging.info(args)

if args.override_target_length is not None:
    target_length = args.override_target_length
    logging.warning(f"target_length = {target_length}")
else:
    target_length = args.target_length_in_k * 1024 + int(args.plus_one)
    logging.warning(f"target_length = {args.target_length_in_k} * 1024 + {int(args.plus_one)} = {target_length}")

if args.batch_size > 0:
    logging.warning(f"batch size = {args.batch_size}, each packing group will contain {args.batch_size} samples")

PAD_TOKEN_ID = args.pad_token_id
IGNORE_INDEX = args.ignore_index
logging.warning(f"PAD_TOKEN_ID: {PAD_TOKEN_ID}")
logging.warning(f"IGNORE_INDEX: {IGNORE_INDEX}")

MODE = args.packing_mode
logging.warning(f"PACKING MODE: {MODE}")

logging.info(f"Loading from {args.input_path}")
ds = load_from_disk(args.input_path)
if args.debug:
    ds = ds.select(range(50000))
logging.info(f"{len(ds)} samples loaded")

if args.process_type == "shuffle":
    logging.warning(f"Shuffling dataset using seed {args.seed}")
    processed_ds = ds.shuffle(seed=args.seed)
else:
    logging.warning(f"Sorting dataset by sequence length")
    processed_ds = ds.sort("real_len")

def pack_samples(sample_indicies: List[int]):
    assert len(sample_indicies) > 0
    logging.debug(f"Packing samples: {sample_indicies}")
    packed_input_ids = np.full(target_length, PAD_TOKEN_ID, dtype=np.int64)
    packed_labels = np.full(target_length, IGNORE_INDEX, dtype=np.int64)
    write_pos = 0
    cu_seqlens = [0]
    cu_seqlens_padded = [0] if MODE == 2 else None
    # max_seqlen = 0
    for i, sample_idx in enumerate(sample_indicies):
        sample = processed_ds[sample_idx]
        if MODE < 2:
            cur_len = cu_seqlens[-1]
        elif MODE == 2:
            cur_len = cu_seqlens_padded[-1]
        else:
            raise NotImplementedError
        allowed_len = target_length - cur_len
        if allowed_len <= 0:
            raise RuntimeError("target_length - cur_len <= 0, This should not happend")
        if allowed_len < sample["real_len"]:
            logging.debug(f"Truncating sample id: {sample_idx}")
        needed_length = None
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        if i == len(sample_indicies) - 1 and sample["real_len"] < allowed_len:
            logging.debug(f"padding the last sample of this batch, id: {sample_idx}")
            needed_length = allowed_len - sample["real_len"]
        input_ids = input_ids[:allowed_len]
        labels = labels[:allowed_len]
        assert len(input_ids) == len(labels)

        copy_len = len(input_ids)
        packed_input_ids[write_pos:write_pos + copy_len] = input_ids
        packed_labels[write_pos:write_pos + copy_len] = labels
        if needed_length is not None:
            new_len = allowed_len
        else:
            new_len = copy_len
        write_pos += new_len

        if MODE == 0:
            cu_seqlens.append(cur_len + new_len)
        elif MODE == 1:
            if needed_length is not None:
                cu_seqlens.append(cur_len + sample["real_len"])
                cur_len = cu_seqlens[-1]
                cu_seqlens.append(cur_len + needed_length)
            else:
                cu_seqlens.append(cur_len + new_len)
        elif MODE == 2:
            if needed_length is not None:
                cu_seqlens.append(cur_len + sample["real_len"])
                cu_seqlens_padded.append(cur_len + new_len)
            else:
                cu_seqlens.append(cur_len + new_len)
                cu_seqlens_padded.append(cur_len + new_len)
        else:
            raise NotImplementedError
    
    assert len(packed_input_ids) == len(packed_labels)
    assert write_pos == target_length, f"write_pos = {write_pos}"
    assert len(packed_input_ids) == target_length, f"len(packed_input_ids) = {len(packed_input_ids)}"
    if MODE == 0:
        assert cu_seqlens[-1] == target_length
        assert len(cu_seqlens) == len(sample_indicies) + 1
    elif MODE == 1:
        assert cu_seqlens[-1] == target_length
        assert len(cu_seqlens) == len(sample_indicies) + 1 or len(cu_seqlens) == len(sample_indicies) + 2
    elif MODE == 2:
        assert cu_seqlens_padded[-1] == target_length
        assert len(cu_seqlens) == len(cu_seqlens_padded)
        assert len(cu_seqlens) == len(sample_indicies) + 1
        assert cu_seqlens[-1] <= cu_seqlens_padded[-1]
        for x, y in zip(cu_seqlens[:-1], cu_seqlens_padded[:-1]):
            assert x == y

    return {
        "input_ids": packed_input_ids,
        "labels": packed_labels,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens_padded,
        "real_len": len(packed_input_ids)
    }

processed_ds_real_lens = processed_ds.select_columns(["real_len"])
assert processed_ds[0]["real_len"] == processed_ds_real_lens[0]["real_len"]
cur_sample_indicies = []
cur_seq_len = 0
packing_groups = []
for i, sample in tqdm(enumerate(processed_ds_real_lens), total=len(processed_ds), desc="Gathering packing groups"):
    if cur_seq_len > 0:
        if cur_seq_len + sample["real_len"] > target_length or (args.batch_size > 0 and len(cur_sample_indicies) == args.batch_size):
            packing_groups.append(cur_sample_indicies)
            # logging.info(cur_sample_indicies)
            # logging.info(cur_seq_len)
            cur_sample_indicies = []
            cur_seq_len = 0
    cur_sample_indicies.append(i)
    cur_seq_len += sample["real_len"]
if cur_seq_len > 0:
    packing_groups.append(cur_sample_indicies)
    cur_sample_indicies = []
    cur_seq_len = 0

logging.info(f"Gathered {len(packing_groups)} packing groups")

def packed_samples_generator():
    for packing_group in tqdm(packing_groups, desc="Packing samples"):
        yield pack_samples(packing_group)

packed_ds = Dataset.from_generator(packed_samples_generator)
logging.info(f"before packing, dataset contains {len(processed_ds)} samples")
logging.info(f"after packing, dataset contains {len(packed_ds)} samples")

packed_ds_name = os.path.basename(args.input_path)
save_dir = os.path.dirname(args.input_path)

if args.override_target_length is not None:
    packed_ds_name = f"{packed_ds_name}_packed_{args.process_type}_mode_{args.packing_mode}_{args.override_target_length}"
else:
    packed_ds_name = f"{packed_ds_name}_packed_{args.process_type}_mode_{args.packing_mode}_{args.target_length_in_k}k"
if args.plus_one:
    packed_ds_name = f"{packed_ds_name}_plus"
if args.process_type == "shuffle":
    packed_ds_name = f"{packed_ds_name}_seed_{args.seed}"
if args.batch_size > 0:
    packed_ds_name = f"{packed_ds_name}_bs_{args.batch_size}"

save_path = f"{save_dir}/{packed_ds_name}"
logging.info(f"Saving packed dataset to {save_path}")
packed_ds.save_to_disk(save_path)
print("done")
