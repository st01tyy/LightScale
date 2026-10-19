import re


def verify_gsm8k_sample(model_output, ground_truth):
    # gsm is easy: extract numbers, and then just compare last number with answer.
    # replace numbers like `x,xxx` with `xxxx`
    response = re.sub(r"(\d),(\d)", r"\1\2", model_output)
    ground_truth = re.sub(r"(\d),(\d)", r"\1\2", ground_truth)

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    if numbers:
        predictions = numbers[-1]
    else:
        predictions = response
    return str(predictions).lower() == str(ground_truth).lower()
