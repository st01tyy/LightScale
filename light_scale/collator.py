from transformers import PreTrainedTokenizer
from typing import List, Tuple, Optional
import math
import torch
from light_scale import dist_utils

# copy from NeMo
def _ceil_to_nearest(n, m, ceil_to_power_2=False):
    if ceil_to_power_2:
        # Reccurent Gemma (AKA Griffin) requires seq length to be a power of 2 for parallel scan
        return 2 ** math.ceil(math.log2(n))
    else:
        return (n + m - 1) // m * m
    
class Collator:
    def __init__(self):
        pass

    def collate_fn(self, samples: list) -> dict:
        raise NotImplementedError

class MultiSamplingActorReferenceCollator(Collator):
    # 适用于多次采样如GRPO，要求每条样本为List[Tuple[prompt, response]]
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 max_length: int,
                 pad_to_multiple_of: Optional[int] = None
                 ):
        super().__init__()
        assert hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = 1
        if pad_to_multiple_of is not None:
            assert max_length % pad_to_multiple_of == 0
            self.pad_to_multiple_of = pad_to_multiple_of

    def collate_fn(self, samples: List[Tuple[str, List[str]]]):
        batch_input_ids = []
        batch_labels = []
        batch_loss_mask = []

        n_samples = len(samples[0])

        for prompt, n_responses in samples:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            for response in n_responses:
                response_ids = self.tokenizer.encode(response, add_special_tokens=False)
                input_ids = prompt_ids + response_ids
                loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
                batch_input_ids.append(input_ids)
                batch_loss_mask.append(loss_mask)
        
        # TODO: variable length
        # max_length = min(self.max_length, _ceil_to_nearest(max([len(input_ids) for input_ids in batch_input_ids]), self.pad_to_multiple_of)) + 1
        max_length = self.max_length + 1

        padded_input_ids = []
        padded_loss_mask = []
        for input_ids, loss_mask in zip(batch_input_ids, batch_loss_mask):
            assert len(input_ids) == len(loss_mask)
            needed = max_length - len(input_ids)
            if needed > 0:
                input_ids += [self.tokenizer.pad_token_id] * needed
                loss_mask += [0] * needed
            padded_input_ids.append(input_ids[:max_length])
            padded_loss_mask.append(loss_mask[:max_length])
        
        batch_input_ids = [input_ids[:-1] for input_ids in padded_input_ids]
        batch_labels = [input_ids[1:] for input_ids in padded_input_ids]
        batch_loss_mask = [loss_mask[1:] for loss_mask in padded_loss_mask]
        
        outputs = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.int64, device=dist_utils.get_device()),
            "labels": torch.tensor(batch_labels, dtype=torch.int64, device=dist_utils.get_device()),
            "loss_mask": torch.tensor(batch_loss_mask, dtype=torch.int64, device=dist_utils.get_device()),
            "n_samples": n_samples
        }

        return outputs
        

class MultiSamplingActorReferencePackingCollator:
    # 适用于多次采样如GRPO，要求每条样本为List[str]
    pass