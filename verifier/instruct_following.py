import re
import json
import string

from verifier.if_utils import IF_FUNCTIONS_MAP


def verify_ifeval_sample(model_output, constraint):
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    if "func_name" not in constraint:
        print("WARNING: constraint missing func_name")
        print(constraint)
        return False
    # first, parse out the constraint string.
    func_name = constraint.pop("func_name")
    # get the function
    func = IF_FUNCTIONS_MAP[func_name]
    # now, run the function
    # pop out any none args
    non_none_args = {k: v for k, v in constraint.items() if v is not None}
    # sometimes we have extra args, sometimes not.
    if len(constraint) == 0:
        return func(model_output)
    return func(model_output, **non_none_args)


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    From https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def verify_flan_sample(model_output, ground_truth_answer):
    # Flan! we will just use... exact match with some basic cleaning, after extracting the answer.
    answer_string = model_output.split("The answer is: ")[-1].strip()
    return normalize_answer(answer_string) == normalize_answer(ground_truth_answer)
