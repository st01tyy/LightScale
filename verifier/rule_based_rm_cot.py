import random

from verifier.format import extract_after_think, verify_format_general
from verifier.math_jiutian import verify_math
from verifier import code_jiutian
# from verifier import sandbox_fusion
from verifier.bash import verify_bash
from verifier.instruct_following import verify_ifeval_sample
from verifier.language import verify_language
from verifier.tool_call_utils import verify_tool_calls_for_qw, verify_tool_calls_for_cm
import json
import traceback


def compute_score(data_source, solution_str, ground_truth, extra_info, **reward_kwargs) -> tuple[float, dict]:
    prompt_str = extra_info

    # 格式奖励
    format_correct = verify_format_general(solution_str)
    format_score = 1.0 if format_correct else 0.0
    final_score = 0.0 if format_correct else 0.0

    correctness_score, language_score = 0.0, 0.0
    tb_str = ""
    if format_correct:
        # 答案正确性奖励
        raw_answer = extract_after_think(solution_str)
        if data_source.lower() in ['tulu_ifeval']:
            answer_correct = verify_ifeval_sample(raw_answer, ground_truth)
        elif data_source.lower() in ['deepscaler', 'gsm8k', 'math', 'deepmath', 'crossthink-math', 'big_math', 'skywork_or1_rl', 'hard_math_rl']:
            try:
                answer_correct = verify_math(raw_answer, ground_truth)
            except Exception as e:
                answer_correct = False
                tb_str = traceback.format_exc()
        elif data_source.lower() in ['tool_calls_cm']:
            answer_correct, correctness_reward = verify_tool_calls_for_cm(raw_answer, ground_truth)
        elif data_source.lower() in ['tool_calls_qw']:
            answer_correct = verify_tool_calls_for_qw(raw_answer, ground_truth)
        elif data_source.lower() in ['code_contests', 'apps', 'taco', 'codeforces', 'leetcode', 'code']:
            url = reward_kwargs.get('code_sandbox_url')
            score = code_jiutian.compute_score(url, raw_answer, ground_truth)
            answer_correct = score > 0.9
        elif data_source.lower() in ['bash']:
            url = reward_kwargs.get('code_sandbox_url')
            answer_correct = verify_bash(url, prompt_str, raw_answer, ground_truth)
        else:
            raise Exception("No parser for such a data_source.")
        correctness_score = 1.0 if answer_correct else 0.0
        final_score = 1.0 if answer_correct else 0.0

        # 语言一致性奖励
        language_correct = verify_language(prompt_str, solution_str)
        language_score = 1.0 if language_correct else 0.0

    reward_metrics = {
        "format": format_score,
        "correctness": correctness_score,
        "language": language_score 
    }

    # with open("/root/work/externalstorage/gpfsprd/taoyuyang/mrl_gkd/async_reward_result.jsonl", mode='a') as f:
    #     f.write(json.dumps(
    #         {"data_source": data_source, "solution_str": solution_str, "ground_truth": ground_truth, "extra_info": extra_info, "tb_str": tb_str, "reward_metrics": reward_metrics},
    #         ensure_ascii=False
    #     ))
    #     f.write('\n')
    #     f.flush()
    return final_score, reward_metrics
