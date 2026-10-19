import random

from verifier.math_jiutian import verify_math
from verifier import code_jiutian
from verifier.bash import verify_bash
from verifier.instruct_following import verify_ifeval_sample
from verifier.tool_call_utils import verify_tool_calls_for_qw, verify_tool_calls_for_cm
from verifier.format import extract_after_think

def check_substring_order(text, first_substring="用户特征分析", second_substring="融合推荐话术"):
    """
    检查文本中是否按顺序包含两个指定的子串。

    参数:
        text (str): 要检查的原始文本。
        first_substring (str): 第一个子串，默认为"用户特征分析"。
        second_substring (str): 第二个子串，默认为"融合推荐话术"。

    返回:
        bool: 如果文本中按顺序（即first_substring在second_substring之前）包含两个子串，则返回True；否则返回False。
    """
    # 查找第一个子串的起始位置
    first_index = text.find(first_substring)
    
    # 如果第一个子串不存在，直接返回False
    if first_index == -1:
        return False
    
    # 从第一个子串之后的位置开始查找第二个子串
    second_index = text.find(second_substring, first_index + len(first_substring))
    
    # 如果第二个子串存在（即找到了），则返回True，否则返回False
    return second_index != -1

def compute_score(data_source, solution_str, ground_truth, extra_info, **reward_kwargs) -> tuple[float, dict]:
    prompt_str = extra_info

    # 如果是慢思考，提取答案部分
    raw_answer = extract_after_think(solution_str)
    if raw_answer is None:
        raw_answer = solution_str

    if data_source.lower() in ['tulu_ifeval']:
        answer_correct = verify_ifeval_sample(raw_answer, ground_truth)
    elif data_source.lower() in ['deepscaler', 'gsm8k', 'math', 'deepmath', 'crossthink-math', 'big_math', 'skywork_or1_rl', 'hard_math_rl']:
        try:
            answer_correct = verify_math(raw_answer, ground_truth)
        except Exception:
            answer_correct = False
    elif data_source.lower() in ['tool_calls_cm']:
        answer_correct, correctness_reward = verify_tool_calls_for_cm(raw_answer, ground_truth)
    elif data_source.lower() in ['tool_calls_qw']:
        answer_correct = verify_tool_calls_for_qw(raw_answer, ground_truth)
    elif data_source.lower() in ['code_contests', 'apps', 'taco', 'codeforces', 'leetcode', 'code']:
        url = reward_kwargs.get('code_sandbox_urls')
        score = code_jiutian.compute_score(url, raw_answer, ground_truth)
        answer_correct = score > 0.9
    elif data_source.lower() in ['bash']:
        url = reward_kwargs.get('code_sandbox_urls')
        answer_correct = verify_bash(url, prompt_str, raw_answer, ground_truth)
    elif data_source.lower() in ['yingxiao']:
        answer_correct = check_substring_order(raw_answer)
    elif data_source.lower() in ['ignore']:
        answer_correct = True
    else:
        raise Exception("No parser for such a data_source.")

    final_score = 1.0 if answer_correct else 0.0
    correctness_score = 1.0 if answer_correct else 0.0
    reward_metrics = {
        "correctness": correctness_score
    }
    # print(f"correctness score: {correctness_score}")
    return final_score, reward_metrics
