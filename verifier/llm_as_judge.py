import requests
import re
import json

import ray
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from verifier.template.simpleQA import GRADER_TEMPLATE
from verifier.format import sperate_query_response, extract_after_think, verify_format_general
from verifier.language import verify_language
from utils.utils import get_stop_token_ids


def postprocess(judgment):
    match = re.search(r"(A|B|C)", judgment)
    grade_letter = match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match
    is_correct = grade_letter == "A"
    score = 1.0 if is_correct else 0.0
    return score


def worker(url, data):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.post(url=url, data=data, headers=headers, timeout=None)
    return response


def remote_async_rm_fn(api_urls, decoded_outputs, ground_truths=None, datasets=None):
    """Submit requests in parallel via the reward model server api. 

    Args:
        api_url: remote reward model api.
        decoded_outputs: the decoded output of model, prompt+response.
        ground_truths: the golden answers to judge queries.
        datasets: the source of the queries.

    Returns:
        scores: the scores of responses.
    """
    judge_model_path = "/gpfsprd/sunjiahui/Qwen2.5-72B-Instruct"
    stop_token_ids = get_stop_token_ids(judge_model_path)
    # 数据预处理
    n_samples = len(decoded_outputs)
    format_correct_list = [False] * n_samples
    language_correct_list = [False] * n_samples
    grader_prompt_list = []
    for i, output in enumerate(decoded_outputs):
        query, response = sperate_query_response(output)
        format_correct = verify_format_general(response)
        language_correct = verify_language(query, response)
        format_correct_list[i] = format_correct
        language_correct_list[i] = language_correct
        if not format_correct:
            continue
        # 只有格式正确的回复，才需要去打分
        answer = extract_after_think(response)
        grader_prompt = GRADER_TEMPLATE.format(
            question=query,
            target=ground_truths[i],
            predicted_answer=answer,
        )
        payload = json.dumps({
            "messages": [
                {
                    "content": grader_prompt,
                    "role": "user"
                }
            ],
            "model": judge_model_path,
            "max_tokens": 1024,
            "stop": None,
            "stream": False,
            "n": 1,
            "temperature": 0.6,
            "top_p": 1,
            "top_k": -1,
            "stop_token_ids": stop_token_ids
        })
        grader_prompt_list.append(payload)

    # 发送请求
    with ThreadPoolExecutor(max_workers=1024) as executor:
        result = [
            executor.submit(
                worker, api_urls[i%len(api_urls)], grader_prompt_list[i]
            ) for i in range(len(grader_prompt_list))
        ]
        # use tqdm to show progress
        for _ in tqdm(as_completed(result), total=len(result)):
            pass
        results = [r.result() for r in result]

    # 打分结果解析
    judgment_scores = [0.0] * len(results)
    for i, result in enumerate(results):
        if result.status_code == 200:
            response_data = json.loads(result.text)
            judgment = response_data["choices"][0]["message"]["content"]
            completion_tokens = response_data["usage"]["completion_tokens"]
            judgment_scores[i] = postprocess(judgment)
            print(f"Judgment score: {judgment_scores[i]}; Judgment tokens: {completion_tokens}")
        else:
            judgment_scores[i] = 0.0
            print(f"Request error code: {result.status_code}")
    correctness_scores = np.zeros(n_samples, dtype=np.float32)
    correctness_scores[format_correct_list] = judgment_scores
    return correctness_scores.tolist()


@ray.remote
def remote_async_rm_fn_ray(api_urls, queries, ground_truths=None, datasets=None):
    return remote_async_rm_fn(api_urls, queries, ground_truths, datasets)


if __name__ == "__main__":
    url = "http://10.17.6.71:8000/v1/chat/completions,http://10.17.6.69:8000/v1/chat/completions,http://10.17.6.70:8000/v1/chat/completions"
    url = url.split(',')
    queries = [
        'User: 美国政治人物德怀特·艾森豪威尔是哪个党派的？. Assistant: <think>究竟是哪个党派呢？</think>国民党',
        'User: 美国政治人物德怀特·艾森豪威尔是哪个党派的？. Assistant: <think>究竟是哪个党派呢？</think>国民党',
        'User: 美国政治人物德怀特·艾森豪威尔是哪个党派的？. Assistant: <think>究竟是哪个党派呢？</think>国民党',
        'User: Who received the IEEE Frank Rosenblatt Award in 2010?. Assistant: <think>Michio</think>Sugeno',
        'User: Who received the IEEE Frank Rosenblatt Award in 2010?. Assistant: Sugeno',
        'User: Who received the IEEE Frank Rosenblatt Award in 2010?. Assistant: Sugeno'
    ]
    ground_truths = [
        '共和党',
        '共和党',
        '共和党',
        'Michio Sugeno',
        'Michio Sugeno',
        'Michio Sugeno'
    ]
    score = remote_async_rm_fn(url, queries, ground_truths)
    print(score)
