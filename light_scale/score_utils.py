from typing import List, Optional
from light_scale.data import Sample, MultiResponseSample, BatchExperience
from verifier.rule_based_rm_cot import compute_score as compute_score_cot
from verifier.rule_based_rm import compute_score
from light_scale import sandbox_fusion_utils
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import numpy as np
from tqdm import tqdm
from megatron.training.global_vars import get_args

def _score_process_fn(dataset_type, response, ground_truth, prompt, force_thinking, use_cot_reward, begin_of_thinking = None, sandbox_fusion_url = None):
    if force_thinking:
        assert begin_of_thinking is not None
        response = begin_of_thinking + response
    code_sandbox_urls = None
    if sandbox_fusion_url is not None:
        code_sandbox_urls=sandbox_fusion_utils.get_run_code_url(sandbox_fusion_url)
    if use_cot_reward:
        reward, reward_metrics = compute_score_cot(dataset_type, response, ground_truth, prompt, code_sandbox_urls=code_sandbox_urls)
    else:
        reward, reward_metrics = compute_score(dataset_type, response, ground_truth, prompt, code_sandbox_urls=code_sandbox_urls)
    return reward, reward_metrics

def _score_thread_fn(
        sample_id: int,
        sample: MultiResponseSample,
        force_thinking: bool, 
        use_cot_reward: bool,
        process_pool: ProcessPoolExecutor,
        begin_of_thinking: str = None,
        alive_urls: List[str] = None,
        url_usages: np.ndarray = None,
        lock: Lock = None
    ):
    prompt = sample.prompt
    responses = sample.responses
    ground_truth = sample.ground_truth
    dataset_type = sample.dataset_type
    rewards = []
    reward_metrics_list = []
    url_id = None
    url = None

    if dataset_type.lower() in ['code_contests', 'apps', 'taco', 'codeforces', 'leetcode', 'code']:
        # 获取代码沙盒url
        assert alive_urls is not None
        assert url_usages is not None and len(url_usages) == len(alive_urls)
        assert lock is not None
        with lock:
            url_id = np.argmin(url_usages)
            url_usages[url_id] += 1
            url = alive_urls[url_id]

    futures = []
    for response in responses:
        if response is None:
            futures.append(None)
            continue
        future = process_pool.submit(_score_process_fn, dataset_type, response, ground_truth, prompt, force_thinking, use_cot_reward, begin_of_thinking, url)
        futures.append(future)
    
    for future, response in zip(futures, responses):
        if future is None:
            assert response is None
            rewards.append(None)
            continue
        reward, reward_metrics = future.result()
        rewards.append(reward)
        reward_metrics_list.append(reward_metrics)

    if url_id is not None:
        with lock:
            url_usages[url_id] -= 1

    sample.rewards = rewards
    sample.reward_metrics_list = reward_metrics_list

    return sample_id, sample

def score(samples: List[MultiResponseSample], num_processes: int = 32) -> List[MultiResponseSample]:
    margs = get_args()
    sandbox_fusion_urls = margs.sandbox_fusion_urls
    alive_urls = None
    url_usages = None
    lock = None
    if sandbox_fusion_urls is not None:
        alive_urls = sandbox_fusion_utils.get_alive_urls(sandbox_fusion_urls)
        assert len(alive_urls) > 0, "No alive sandbox fusion service"
        url_usages = np.zeros((len(alive_urls),), dtype=np.int32)
        lock = Lock()
    thread_pool = ThreadPoolExecutor(max_workers=len(samples))
    num_processes = 5 * len(alive_urls) if alive_urls is not None else num_processes
    process_pool = ProcessPoolExecutor(max_workers=num_processes)
    futures = []
    for i, sample in tqdm(enumerate(samples), desc="Submit score", total=len(samples)):
        future = thread_pool.submit(_score_thread_fn, i, sample, margs.force_thinking, margs.use_cot_reward, process_pool, margs.begin_of_thinking, alive_urls, url_usages, lock)
        futures.append(future)
    for future in tqdm(as_completed(futures), desc="Score", total=len(futures)):
        future.result()
    thread_pool.shutdown()
    process_pool.shutdown()
    return samples