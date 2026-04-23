import socket
import sys
import psutil
from typing import Set

def get_local_identifiers() -> Set[str]:
    """
    获取本地主机的所有可能标识符，使用 psutil 库以获得最佳可靠性。
    
    这包括：
    1. 主机名和 FQDN (来自 socket 模块)。
    2. 所有网络接口上的所有非环回 IPv4 和 IPv6 地址 (来自 psutil)。

    :return: 一个包含所有标识符的集合 (set)。
    """
    identifiers = set()
    
    # 1. 获取主机名和 FQDN
    try:
        hostname = socket.gethostname()
        identifiers.add(hostname)
        fqdn = socket.getfqdn()
        identifiers.add(fqdn)
        print(f"--- 发现主机名: '{hostname}', FQDN: '{fqdn}' ---", file=sys.stderr)
    except Exception as e:
        print(f"--- 警告: 获取主机名时出错: {e} ---", file=sys.stderr)

    # 2. 使用 psutil 获取所有网络接口的IP地址 (核心优化)
    print("--- 使用 psutil 扫描网络接口... ---", file=sys.stderr)
    found_ips = set()
    
    # psutil.net_if_addrs() 返回一个字典，键是接口名称，值是地址列表
    all_interfaces = psutil.net_if_addrs()
    for interface_name, interface_addresses in all_interfaces.items():
        for addr in interface_addresses:
            # AF_INET 表示 IPv4, AF_INET6 表示 IPv6
            if addr.family == socket.AF_INET or addr.family == socket.AF_INET6:
                # addr.address 就是我们需要的 IP 地址字符串
                ip_address = addr.address
                # 过滤掉环回地址 (e.g., '127.0.0.1' or '::1')
                if not ip_address.startswith('127.') and not ip_address == '::1':
                     # 过滤掉本地链接地址 (fe80::)，这些地址通常不用于主机间通信
                    if not ip_address.startswith('fe80:'):
                        found_ips.add(ip_address)
                        print(f"    > 在接口 '{interface_name}' 上发现IP: {ip_address}", file=sys.stderr)

    print(f"--- 发现的IP地址: {found_ips} ---", file=sys.stderr)
    identifiers.update(found_ips)

    print(f"--- 用于匹配的本地标识符: {identifiers} ---", file=sys.stderr)
    
    return identifiers

def get_node_rank(hosts_in_file: list) -> int:
    """
    确定当前主机排序
    """
    local_identifiers = get_local_identifiers()

    for i, host_entry in enumerate(hosts_in_file):
        if host_entry in local_identifiers:
            return i

    return -1