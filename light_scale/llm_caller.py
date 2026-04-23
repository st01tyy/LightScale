from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from openai import OpenAI
from typing import List, Tuple
import json
import requests
from light_scale.logger_utils import setup_logger
from light_scale.config import RolloutServiceConfig, ReferenceModelConfig
from requests.exceptions import Timeout, RequestException
import time
from tqdm import tqdm
from openai import BadRequestError
import queue
import re

def wait_until_rollout_and_ref_server_ready(rollout_service_config: RolloutServiceConfig, reference_config: ReferenceModelConfig, retry=5, interval=10):
    """
    等待直到两个服务都就绪或达到最大重试次数
    
    :param rollout_service_config: actor服务配置
    :param reference_config: ref服务配置
    :param retry: 最大重试次数，默认5次
    :param interval: 重试间隔时间(秒)，默认10秒
    :return: 是否成功等到服务就绪
    """
    logger = setup_logger("light_scale")

    def is_rollout_and_ref_server_ready(rollout_service_config: RolloutServiceConfig, reference_config: ReferenceModelConfig):
        # 检查 rollout 服务
        rollout_ready = False
        for rollout_base_url in rollout_service_config.rollout_base_url_list:
            try:
                rollout_health_url = rollout_base_url + "/health"
                response = requests.get(rollout_health_url, timeout=30)
                rollout_ready = response.status_code == 200
            except Timeout:
                logger.info(f"Rollout 服务健康检查超时: {rollout_health_url}")
                rollout_ready = False
            except RequestException as e:
                logger.info(f"Rollout 服务健康检查请求异常: {e}")
                rollout_ready = False
            
            if not rollout_ready:
                return False
        
        # 检查 ref 服务
        if reference_config is None or reference_config.reference_service_url is None:
            return True
        ref_ready = False
        try:
            ref_server_url = reference_config.reference_service_url
            response = requests.get(ref_server_url, timeout=30)
            ref_ready = response.status_code == 200
        except Timeout:
            logger.info(f"Ref 服务健康检查超时: {ref_server_url}")
            ref_ready = False
        except RequestException as e:
            logger.info(f"Ref 服务健康检查请求异常: {e}")
            ref_ready = False
        
        return rollout_ready and ref_ready

    for attempt in range(1, retry + 1):
        logger.info(f"尝试检查服务状态 ({attempt}/{retry})...")
        
        if is_rollout_and_ref_server_ready(rollout_service_config, reference_config):
            logger.info("所有服务已就绪!")
            return True
        
        if attempt < retry:
            logger.info(f"服务尚未就绪，等待 {interval} 秒后重试...")
            time.sleep(interval)
    
    logger.warning(f"达到最大重试次数 {retry}，服务仍未就绪")
    return False

class ReferenceModelCaller:
    def __init__(self, url: str):
        self.url = url
        self.thread_pool = ThreadPoolExecutor(1)
        self.logger = setup_logger("light_scale")

    def batch_logp(self, samples: list, logger) -> list:
        if self.url == "debug":
            return "debug" # for debug
        payload = json.dumps(samples)
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        response = requests.request("POST", self.url + '/call', headers=headers, data=payload, timeout=None)
        if response.status_code != 200:
            logger.error(response.text)
            raise RuntimeError(f"status code: {response.status_code}")
        response_data = json.loads(response.text)["response"]
        logger.info(f"recevied {len(response_data)} ref logp samples")
        return response_data

    def async_batch_logp(self, samples: list) -> Future:
        self.logger.info(f"async_batch_logp: {len(samples)} samples")
        return self.thread_pool.submit(self.batch_logp, samples, self.logger)

class InferenceServiceCaller:
    # 调用模型推理服务
    def __init__(self, url_list: List[str], model_name: str, batch_size: int = None, num_workers: int = 4000):
        self.url_list = url_list
        self.model_name = model_name
        self.thread_pool = ThreadPoolExecutor(num_workers)
        clients = []
        for url in url_list:
            clients.append(OpenAI(base_url=url, api_key="None"))
        self.clients = clients
        self.batch_size = batch_size
        self.logger = setup_logger("light_scale")
        self.first_call = True

    def _call(self, prompt_id: int, prompt: str, n_samples: int, add_stop: bool, sampling_params: dict, client_id: int = None):
        if client_id is None:
            client_id = prompt_id % len(self.clients)
        self.logger.debug(f"using client id: {client_id}")
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            url = self.url_list[client_id]
            effective_sampling_params = dict(sampling_params)
            has_retried_with_reduced_max_tokens = False
            while True:
                payload = json.dumps({
                    "prompt": prompt,
                    "model": self.model_name,
                    "stream": False,
                    "n": n_samples,
                    **effective_sampling_params
                })
                response = requests.request("POST", f"{url}/completions", headers=headers, data=payload, timeout=3600)
                if response.status_code == 200:
                    response_data = json.loads(response.text)
                    choices = response_data["choices"]
                    completion_tokens = response_data["usage"]["completion_tokens"]
                    total_tokens = response_data["usage"]["total_tokens"]
                    response_texts = []
                    for choice in choices:
                        response_text = choice["text"]
                        finish_reason = choice["finish_reason"]
                        if finish_reason == "stop" and "stop" in sampling_params and add_stop:
                            stop = sampling_params.get("stop")
                            if not response_text.endswith(stop):
                                response_text += stop
                        response_texts.append(response_text)
                    assert len(response_texts) == n_samples
                    return prompt_id, response_texts, completion_tokens, total_tokens, client_id

                adjusted_max_tokens = self._get_adjusted_max_tokens_from_bad_request(
                    response_status_code=response.status_code,
                    response_text=response.text,
                    sampling_params=effective_sampling_params,
                )
                if adjusted_max_tokens is not None and not has_retried_with_reduced_max_tokens:
                    old_max_tokens = effective_sampling_params.get("max_tokens")
                    effective_sampling_params["max_tokens"] = adjusted_max_tokens
                    has_retried_with_reduced_max_tokens = True
                    self.logger.warning(
                        f"prompt_id={prompt_id} hit max context length; adjust max_tokens from {old_max_tokens} to {adjusted_max_tokens} and retry once"
                    )
                    continue

                raise RuntimeError(f"status code: {response.status_code}, response: {response.text}")
        except Exception as e:
            self.logger.error(str(e))
            return prompt_id, [None] * n_samples, 0, 0, client_id

    def batch_completions(self, prompts: List, n_samples: int, add_stop: bool = False, **sampling_params) -> List[Tuple[List[str], int]]:
        if self.first_call:
            self.logger.warning(f"sampling params: {sampling_params}")
            self.first_call = False
        if self.batch_size is not None:
            return self._batch_size_specified_completions(prompts, n_samples, add_stop, sampling_params)
        futures = [self.thread_pool.submit(self._call, id, prompt, n_samples, add_stop, sampling_params) for id, prompt in enumerate(prompts)]
        res = [future.result() for future in tqdm(as_completed(futures), desc="Generating responses", total=len(futures))]
        res = sorted(res, key=lambda x: x[0])
        return [(r[1], r[2], r[3]) for r in res]

    def _get_adjusted_max_tokens_from_bad_request(self, response_status_code: int, response_text: str, sampling_params: dict):
        if response_status_code != 400:
            return None

        try:
            response_json = json.loads(response_text)
            err_msg = response_json.get("message", "")
        except Exception:
            err_msg = response_text

        # Example:
        # Requested token count exceeds the model's maximum context length of 32768 tokens.
        # You requested a total of 33043 tokens: 3043 tokens from the input messages and
        # 30000 tokens for the completion.
        match = re.search(
            r"maximum context length of\s+(\d+)\s+tokens.*?requested a total of\s+(\d+)\s+tokens:\s+(\d+)\s+tokens from the input messages and\s+(\d+)\s+tokens for the completion",
            err_msg,
            re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            return None

        max_context_len = int(match.group(1))
        input_tokens = int(match.group(3))
        requested_completion_tokens = int(match.group(4))

        safe_max_tokens = max_context_len - input_tokens - 100
        if safe_max_tokens <= 0:
            return None

        current_max_tokens = sampling_params.get("max_tokens", requested_completion_tokens)
        if not isinstance(current_max_tokens, int):
            return safe_max_tokens
        if safe_max_tokens >= current_max_tokens:
            return None
        return safe_max_tokens
    
    def _batch_size_specified_completions(self, prompts: List, n_samples: int, add_stop: bool, sampling_params: dict) -> List[Tuple[List[str], int]]:
        batch_size = self.batch_size
        assert batch_size is not None and batch_size > 0

        prompt_id = 0
        sample_id = 0
        pbar = tqdm(desc="Generating responses", initial=0, total=len(prompts) * n_samples)

        # 初始提交
        cur_futures = []
        for i in range(batch_size):
            if prompt_id == len(prompts):
                break
            future = self.thread_pool.submit(
                self._call, 
                prompt_id * n_samples + sample_id, prompts[prompt_id], 1, add_stop, sampling_params
            )
            cur_futures.append(future)
            sample_id += 1
            if sample_id == n_samples:
                prompt_id += 1
                sample_id = 0
        
        # 等待结果&负载均衡接续提交
        next_futures = []
        raw_results = []
        while len(cur_futures) > 0:
            for future in as_completed(cur_futures):
                raw_res = future.result()
                client_id = raw_res[-1]
                raw_results.append(raw_res)
                pbar.update()
                if prompt_id < len(prompts):
                    next_future = self.thread_pool.submit(
                        self._call, 
                        prompt_id * n_samples + sample_id, prompts[prompt_id], 1, add_stop, sampling_params, client_id
                    )
                    next_futures.append(next_future)
                    sample_id += 1
                    if sample_id == n_samples:
                        prompt_id += 1
                        sample_id = 0
            
            cur_futures = next_futures
            next_futures = []
        pbar.close()

        # 处理结果
        assert len(raw_results) == len(prompts) * n_samples
        raw_results = sorted(raw_results, key=lambda x: x[0])
        results = []
        for i in range(0, len(raw_results), n_samples):
            group_results = raw_results[i:i+n_samples]
            responses = [r[1][0] for r in group_results]
            completion_tokens = sum([r[2] for r in group_results])
            total_tokens = sum([r[3] for r in group_results])
            results.append((responses, completion_tokens, total_tokens))

        return results
    
    def __del__(self):
        self.thread_pool.shutdown()