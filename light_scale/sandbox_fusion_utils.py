import requests
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

def ping_sandbox(base_url: str) -> bool:
    payload = {}
    headers = {
        'Accept': 'application/json'
    }
    try:
        response = requests.request("GET", f"{base_url}/v1/ping", headers=headers, data=payload, timeout=5)
        return response.status_code == 200, base_url
    except Exception:
        return False, base_url

def get_alive_urls(base_url_list: List[str]) -> List[str]:
    alive_urls = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(ping_sandbox, url) for url in base_url_list]
        for future in as_completed(futures):
            res, base_url = future.result()
            if res:
                alive_urls.append(base_url)
    return alive_urls

def init_sandbox_fusion_urls_from_hostfile(hostfile: str):
    base_urls = []
    with open(hostfile) as f:
        for raw_line in f:
            base_urls.append(raw_line.strip())
    alive_urls = get_alive_urls(base_urls)
    assert len(alive_urls) > 0, "No alive sandbox fusion service"
    return base_urls

def get_run_code_url(base_url: str):
    return f"{base_url}/run_code"