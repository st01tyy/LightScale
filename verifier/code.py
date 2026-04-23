import requests
from concurrent.futures import ThreadPoolExecutor
import json
from typing import List


def worker(url: str, data: str, timeout: int = 30) -> requests.Response:
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.post(url, data=data, headers=headers, timeout=timeout)
    return response


def verify_code(url, question, response, ground_truth) -> bool:
    test_cases = json.loads(ground_truth)
    test_cases = [
        {
            'input': {
                'stdin': stdin
            },
            'output': {
                'stdout': stdout
            }
        } for stdin, stdout in zip(test_cases['inputs'], test_cases['outputs'])
    ]
    ground_truth = {
        "id": 0,
        "content": question,
        "labels": {
            "problem_format": "Standard",
        },
        "test": test_cases,
        "canonical_solution": ''
    }
    data = json.dumps({
        'dataset': 'dataset_id',
        'id': 0,
        'config': {
            'dataset_type': 'CommonOJDataset',
            'provided_data': ground_truth,
            'language': 'python',
        },
        'completion': response,
    }, allow_nan=True)
    try:
        result = worker(url, data)
        result.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Code Sandbox请求出错: ", e)
        print(f"{question=}")
        return False
    if result.status_code == 200:
        test_cases = result.json()['tests']
        return all([res['passed'] for res in test_cases])
    else:
        return False


def verify_code_batch(api_urls, questions, responses, ground_truths) -> List[bool]:
    # 数据准备
    json_data_list = []
    for i, (question, response, ground_truth) in enumerate(zip(questions, responses, ground_truths)):
        test_cases = json.loads(ground_truth)
        test_cases = [
            {
                'input': {
                    'stdin': stdin
                },
                'output': {
                    'stdout': stdout
                }
            } for stdin, stdout in zip(test_cases['inputs'], test_cases['outputs'])
        ]
        ground_truth = {
            "id": i,
            "content": question,
            "labels": {
                "problem_format": "Standard",
            },
            "test": test_cases,
            "canonical_solution": ''
        }
        data = json.dumps({
            'dataset': 'dataset_id',
            'id': i,
            'config': {
                'dataset_type': 'CommonOJDataset',
                'provided_data': ground_truth,
                'language': 'python',
            },
            'completion': response,
        }, allow_nan=True)
        json_data_list.append(data)

    # 发送请求
    with ThreadPoolExecutor(max_workers=100) as executor:
        results = [
            executor.submit(
                worker, api_urls[i%len(api_urls)], json_data_list[i]
            ) for i in range(len(json_data_list))
        ]
        results = [r.result() for r in results]

    # 结果解析
    final_res = [False] * len(results)
    for i, result in enumerate(results):
        if result.status_code == 200:
            test_cases = result.json()['tests']
            final_res[i] = all([res['passed'] for res in test_cases])
    return final_res


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import time

    api_urls = 'http://10.17.6.8:8080/submit,http://10.17.6.9:8080/submit,http://10.17.6.10:8080/submit,http://10.17.6.11:8080/submit,http://10.17.6.23:8080/submit'
    api_urls = api_urls.split(',')
    df = pd.read_json("/llmcapagroup1/sft/sunjiahui/RLHF/OpenRLHF/output/code_df.jsonl", lines=True)[:100]
    # results = []
    # for i, row in tqdm(df.iterrows()):
    #     res = verify_code(api_urls[i%len(api_urls)], row['problem'], row['deepseek_solution'], row['ground_truth'])
    #     results.append(res)
    st = time.time()
    results = verify_code_batch(api_urls, df.problem.to_list(), df.deepseek_solution.to_list(), df.ground_truth.to_list())
    print(f"Cost: {time.time()-st:.2f}")
    print(f"Acc: {np.mean(results)}")
