import subprocess
import threading
import signal
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def run_commands_in_pool(cmd_list):
    """
    使用线程池并行运行命令，并在出现错误或接收到信号时终止所有进程。

    Args:
        cmd_list: 要运行的命令列表，每个命令都是一个字符串列表。

    Returns:
        如果所有命令都成功运行，则返回 0。
        如果任何命令失败或接收到信号，则返回 1。
    """

    # 用于存储所有子进程的列表
    processes = []
    # 用于在信号处理程序中设置的标志，以指示是否应该终止进程
    kill_now = False
    # 用于在任何子进程失败时设置的标志
    error_occurred = False
    # 用于线程间同步
    lock = threading.Lock()

    def signal_handler(sig, frame):
        nonlocal kill_now
        print(f"接收到信号 {sig}，正在尝试终止所有进程...")
        with lock:
            kill_now = True
            print(f"接收到信号 {sig}，正在终止所有进程...")
            terminate_all_processes()
            sys.exit(1)

    def terminate_all_processes():
        with lock:
            for p in processes:
                try:
                    p.terminate()
                except OSError:
                    pass  # 进程可能已经结束
                finally:
                    try:
                        p.wait(30)  # 等待 30 秒
                    except subprocess.TimeoutExpired:
                        print("进程超时，发送 SIGKILL 信号...")
                        p.kill()
                    p.wait() # 确保清理僵尸进程

    # 注册信号处理程序
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def run_command(cmd):
        nonlocal error_occurred, kill_now
        process = None  # 初始化 process 变量
        with lock:
            if kill_now or error_occurred:
                return 1  # 如果已经收到信号或发生错误，则不启动新命令
            print(f"正在运行命令: {cmd}")
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    text=True  # 确保输出是文本
                )
                processes.append(process)
            except Exception as e:
                with lock:
                    print(f"启动命令时发生错误: {e}")
                    error_occurred = True
                return 1

        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                if kill_now:
                    break

            process.wait(30)  # 等待 30 秒
            return process.returncode

        except subprocess.TimeoutExpired:
            with lock:
                print("进程超时，发送 SIGKILL 信号...")
                process.kill()
                error_occurred = True
            return 1
        except Exception as e:
            with lock:
                print(f"运行命令时发生错误: {e}")
                error_occurred = True
            return 1
        finally:
            with lock:
                if process in processes:
                    processes.remove(process)  # 确保进程从列表中删除
                if process:
                    try:
                        process.stdout.close()
                    except:
                        pass
                    if process.poll() is None:  # 检查进程是否仍在运行
                        try:
                            process.terminate()  # 确保进程被终止
                        except OSError:
                            pass
                    try:
                        process.wait(30)  # 等待 30 秒
                    except subprocess.TimeoutExpired:
                        print("进程超时，发送 SIGKILL 信号...")
                        process.kill()
                    process.wait() # 确保清理僵尸进程


    with ThreadPoolExecutor(max_workers=len(cmd_list)) as executor:
        futures = [executor.submit(run_command, cmd) for cmd in cmd_list]

        for future in as_completed(futures):
            return_code = future.result()
            with lock:
                if return_code != 0:
                    error_occurred = True
                    print(f"命令失败，返回码: {return_code}")
                    terminate_all_processes()
                    break  # 退出循环

    if error_occurred or kill_now:
        return 1
    else:
        return 0

def run_commands(cmd_list):
    exit_code = run_commands_in_pool(cmd_list)
    sys.exit(exit_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd-file", required=True)
    args = parser.parse_args()
    cmd_list = []
    with open(args.cmd_file) as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 0:
                cmd_list.append(line)
    run_commands(cmd_list)
    