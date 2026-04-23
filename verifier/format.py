import re
from typing import Tuple


THINK_TAG_LIST = [('<think>', '</think>'), ('<|begin_of_thought|>', '<|end_of_thought|>'), ('<THINK>', '</THINK>')]


def extract_after_think(processed_str: str) -> str:
    """Extracts the content after the <think>...</think> block. Return None if </think> is not found."""
    for think_start, think_end in THINK_TAG_LIST:
        end_pos = processed_str.find(think_end)
        if end_pos != -1:
            return processed_str[end_pos+len(think_end):].strip()
    return None


def sperate_query_response(raw_output: str) -> Tuple[str, str]:
    r1_distill_pattern = r"<｜User｜>(.*?)<｜Assistant｜>(.*)"
    r1_base_pattern = r"User:(.*?)Assistant:(.*)"
    chatml_pattern = r"user\n(.*?)assistant\n(.*)"
    pattern_list = [r1_distill_pattern, r1_base_pattern, chatml_pattern]

    for pattern in pattern_list:
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            user_query = match.group(1).strip()
            assistant_response = match.group(2).strip()
            return user_query, assistant_response

    print("Warning!!! Sperating query and response failed for the following output:")
    print(raw_output)
    return '', raw_output


def verify_format(processed_str: str, think_start: str = '<think>', think_end: str = '</think>') -> bool:
    """Validates that the processed string contains a properly formatted <think>...</think> block."""
    validation_passed = True

    # Define required tags
    tags = {
        'think_start': (think_start, 1),
        'think_end': (think_end, 1)
    }

    # Verify tag number
    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = processed_str.find(tag_str)
        if count != expected_count:
            validation_passed = False

    # Verify tag order
    if positions['think_start'] > positions['think_end']:
        validation_passed = False

    return validation_passed


def verify_format_general(processed_str: str) -> bool:
    results = [verify_format(processed_str, think_start, think_end) for think_start, think_end in THINK_TAG_LIST]
    return any(results)
