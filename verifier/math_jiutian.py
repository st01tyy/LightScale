from typing import Optional
import re
import logging
import sys
import os
import contextlib

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


# logging.basicConfig(level=logging.INFO)
logging.getLogger("math_verify").setLevel(logging.CRITICAL + 1)
logging.getLogger("math_verify.metric").setLevel(logging.CRITICAL + 1)
# logging.getLogger("math_verify").setLevel(logging.INFO)
# logging.getLogger("math_verify.metric").setLevel(logging.INFO)

@contextlib.contextmanager
def suppress_all_output():
    # 打开 null 设备，丢弃所有输出
    yield
    return
    # with open(os.devnull, 'w') as devnull:
    #     # 保存当前状态
    #     # old_stdout = sys.stdout
    #     # old_stderr = sys.stderr
    #     # old_logging_level = logging.getLogger().level

    #     try:
    #         # 重定向 stdout/stderr
    #         # sys.stdout = devnull
    #         # sys.stderr = devnull

    #         # 设置 root logger 的级别为最高，屏蔽所有日志
    #         # logging.getLogger().setLevel(logging.CRITICAL + 1)
    #         # 也可指定具体 logger，如 math_verify
    #         # logging.getLogger("math_verify").setLevel(logging.CRITICAL + 1)
    #         # logging.getLogger("math_verify.metric").setLevel(logging.CRITICAL + 1)

    #         yield  # 执行上下文代码块
    #     finally:
    #         # 恢复
    #         pass
    #         # sys.stdout = old_stdout
    #         # sys.stderr = old_stderr
    #         # logging.getLogger().setLevel(old_logging_level)


def grade_answer_mathVerify(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        with suppress_all_output():
            ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        ret_score = timeout_score
    except Exception as e:
        logging.exception(f"[Math-verify failed]:\n{model_output=}\n{ground_truth=}")
        pass

    return ret_score > 0.5


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{" in string:
        splits = string.split("\\text{")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not len(split):
            return string
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def normalize_string(string: str) -> str:
    if string is None:
        return None
    string = str(string).strip()
    if not string:  # ' ', '\n'
        return string

    # ==== 移除/替换 ====
    string = _remove_right_units(string)
    string = re.sub(r"(\\text\{)(.*?)(\})", "\\2", string)  # 去掉 \text{} 公式，但会保留其他内容
    string = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", string)
    string = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", string)
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\$", "").replace("\$", "").replace("$", "")
    string = string.replace("\\%", "").replace("\%", "").replace("%", "")
    string = string.replace(" or ", " , ")
    string = string.replace(" and ", " , ")
    string = re.sub(r",\\! *", "", string)  # 移除逗号+反空格
    # 移除单位
    string = re.sub(r"\^\s*(\\circ|\{\\circ\})", "", string)  # 移除所有角度符号，如 ^\circ、^{\circ}、^ {\circ}
    units = [
        "degree", "mph", "km", "cm", "mm", "centimeter", "meter", "mile", "second",
        "minute", "hour", "day", "week", "month", "year", "foot", "feet",
        "inch", "yard", "liter", "dollar", "cent", "pound", "pm", "am",
        "million", "billion", "trillion"
    ]
    for unit in units:
        string = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", string)

    string = string.strip()
    if len(string) > 0 and string[0] == "{" and string[-1] == "}":
        string = string[1:-1].strip()
    # 移除简单的变量赋值，e.g. "k = " or "q = "
    if "=" in string:
        parts = string.split("=")
        if len(parts) == 2 and len(parts[0].strip()) <= 2:
            string = parts[1].strip()
    # 修复小数
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    # 处理整数
    if _is_float(string) and _is_int(float(string)):
        string = str(int(round(float(string))))
    if _str_is_int(string):  # 1,000,000 -> 1000000
        string = str(_str_to_int(string))
    string = _inject_implicit_mixed_number(string)
    string = _fix_sqrt(string)
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    # string = string.replace("{", "").replace("}", "")
    string = string.replace(" ", "")
    string = string.lower()
    return string


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def remove_equal_sign(given_answer: str, ground_truth: str) -> list[str, str]:
    if '=' in given_answer and '=' not in ground_truth:
        given_answer = given_answer.split('=')[-1]
    if '=' not in given_answer and '=' in ground_truth:
        ground_truth = ground_truth.split('=')[-1]
    return given_answer.strip(), ground_truth.strip()


def compare_strings_commas(given_answer: str, ground_truth: str) -> bool:
    """处理因顺序不同而判错的case"""
    pred_cnt = given_answer.count(',')
    gt_cnt = ground_truth.count(',')
    if pred_cnt != gt_cnt or pred_cnt == 0:
        return False
    else:
        pred = set([x.strip() for x in given_answer.split(',') if len(x)])
        gt = set([x.strip() for x in ground_truth.split(',') if len(x)])
        return pred == gt


def grade_answer_normalize(pred_answer: str, ground_truth: str) -> bool:
    try:
        # normalization
        pred_answer_normalized = normalize_string(pred_answer)
        ground_truth_normalized = normalize_string(ground_truth)
        if pred_answer_normalized == ground_truth_normalized:
            return True

        # 如果一方有=号，另一方没有等号，去掉等号比较一下试试
        pred_answer_normalized, ground_truth_normalized = \
            remove_equal_sign(pred_answer_normalized, ground_truth_normalized)
        if pred_answer_normalized == ground_truth_normalized:
            return True

        # 如果有多个逗号，按逗号切分比较一下两个集合
        if compare_strings_commas(pred_answer_normalized, ground_truth_normalized):
            return True
    except Exception as e:
        logging.exception(f"[Normalize failed]:\n{pred_answer=}\n{ground_truth=}")
        pass
    return False


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = "\\boxed" + string[left_brace_idx:right_brace_idx + 1]
    
    return retval


def remove_boxed(s: str) -> Optional[str]:
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1].strip()
    except:
        return None


def extract_answer(model_output: str) -> str:
    boxed_answer = last_boxed_only_string(model_output)
    if boxed_answer is not None:
        return remove_boxed(boxed_answer)
    return model_output


def verify_math(model_output: str, ground_truth: str) -> bool:
    if not ground_truth or not model_output:
        return False
    ground_truth = str(ground_truth)
    if '\\boxed' in ground_truth:
        ground_truth = extract_answer(ground_truth)
    pred_answer = extract_answer(model_output)
    return pred_answer == ground_truth or \
        grade_answer_normalize(pred_answer, ground_truth) or \
        grade_answer_mathVerify(model_output, ground_truth)
