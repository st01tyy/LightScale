import socket
import threading
import collections
import logging
import multiprocessing

from light_scale.logger_utils import setup_logger

class LockServer:
    """
    一个基于 TCP Socket 的简单分布式锁服务器。
    """
    def __init__(self, ready_event, host='127.0.0.1', port=9999, log_level=logging.INFO):
        self.host = host
        self.port = port
        # self.locks 存储当前被持有的锁以及持有者信息
        # 格式: {'lock_name': client_address}
        self.locks = {}
        # self.waiting_clients 存储等待某个锁的客户端连接
        # 格式: {'lock_name': deque([client_socket1, client_socket2])}
        self.waiting_clients = collections.defaultdict(collections.deque)
        # 服务器内部的线程锁，用于保护对 self.locks 和 self.waiting_clients 的访问
        self.server_lock = threading.Lock()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置地址重用，以便服务器重启后能立即绑定端口
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.logger = setup_logger("dist_lock", level=log_level)

        self.ready_event = ready_event

    def handle_client(self, client_socket, client_address):
        """
        在单独的线程中处理每个客户端的连接。
        """
        self.logger.debug(f"[+] 新的连接来自: {client_address}")
        try:
            while True:
                # 接收客户端的命令
                request = client_socket.recv(1024).decode('utf-8')
                if not request:
                    # 客户端断开连接
                    break
                
                parts = request.strip().split()
                command = parts[0].upper()
                
                if len(parts) < 2:
                    client_socket.sendall(b'ERROR: Invalid command format. Use: COMMAND lock_name')
                    continue
                    
                lock_name = parts[1]

                if command == 'ACQUIRE':
                    self.handle_acquire(client_socket, client_address, lock_name)
                elif command == 'RELEASE':
                    self.handle_release(client_socket, client_address, lock_name)
                else:
                    client_socket.sendall(b'ERROR: Unknown command')

        except ConnectionResetError:
            self.logger.debug(f"[-] 与 {client_address} 的连接被重置。")
        finally:
            # 客户端断开连接后的清理工作
            self.logger.debug(f"[-] 连接关闭: {client_address}")
            self.cleanup_client(client_socket, client_address)
            client_socket.close()

    def handle_acquire(self, client_socket, client_address, lock_name):
        with self.server_lock:
            # 检查锁是否已经被持有
            if lock_name not in self.locks:
                # 锁是可用的，分配给客户端
                self.locks[lock_name] = client_address
                self.logger.debug(f"[LOCK] 锁 '{lock_name}' 被 {client_address} 获取。")
                client_socket.sendall(b'ACQUIRED')
            else:
                # 锁被持有，将客户端加入等待队列
                self.waiting_clients[lock_name].append(client_socket)
                self.logger.debug(f"[WAIT] {client_address} 正在等待锁 '{lock_name}'。")
                # 注意：此处不发送任何消息，客户端的 recv() 会阻塞，直到锁被释放并分配给它

    def handle_release(self, client_socket, client_address, lock_name):
        with self.server_lock:
            # 检查锁是否由请求释放的客户端持有
            if self.locks.get(lock_name) == client_address:
                del self.locks[lock_name]
                self.logger.debug(f"[UNLOCK] 锁 '{lock_name}' 被 {client_address} 释放。")
                client_socket.sendall(b'RELEASED')

                # 检查是否有等待该锁的客户端
                if lock_name in self.waiting_clients and self.waiting_clients[lock_name]:
                    # 从等待队列中取出下一个客户端
                    next_client_socket = self.waiting_clients[lock_name].popleft()
                    next_client_address = next_client_socket.getpeername()
                    
                    # 将锁分配给下一个客户端
                    self.locks[lock_name] = next_client_address
                    self.logger.debug(f"[LOCK] 锁 '{lock_name}' 被自动分配给等待中的 {next_client_address}。")
                    next_client_socket.sendall(b'ACQUIRED')
            else:
                # 如果尝试释放一个不属于自己的锁
                client_socket.sendall(b'ERROR: You do not hold this lock')

    def cleanup_client(self, client_socket, client_address):
        """
        当客户端意外断开时，释放它可能持有的所有锁。
        """
        with self.server_lock:
            # 查找并释放该客户端持有的所有锁
            locks_to_release = [lock for lock, holder in self.locks.items() if holder == client_address]
            for lock_name in locks_to_release:
                self.logger.debug(f"[CLEANUP] 客户端 {client_address} 断开，自动释放锁 '{lock_name}'。")
                # 直接调用释放逻辑，以通知等待者
                self.handle_release(client_socket, client_address, lock_name)
            
            # 从所有等待队列中移除该客户端
            for lock_name in self.waiting_clients:
                queue = self.waiting_clients[lock_name]
                if client_socket in queue:
                    queue.remove(client_socket)

    def start(self):
        """
        启动锁服务器。
        """
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.logger.debug(f"[*] 分布式锁服务器正在监听 {self.host}:{self.port}")

        if self.ready_event:
            self.ready_event.set()

        while True:
            client_socket, client_address = self.server_socket.accept()
            # 为每个客户端创建一个新线程来处理
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_handler.start()

class LockServerProcess(multiprocessing.Process):
    """
    将 LockServer 封装在一个独立的进程中，并提供带有启动超时机制的上下文管理器。
    """
    def __init__(self, host='127.0.0.1', port=9999, startup_timeout=10): # 1. 添加超时参数
        super().__init__()
        self.host = host
        self.port = port
        self.daemon = True
        self.startup_timeout = startup_timeout
        self.ready_event = multiprocessing.Event()

    def run(self):
        lock_server = LockServer(
            host=self.host, 
            port=self.port, 
            ready_event=self.ready_event
        )
        lock_server.start()

    def start_lock_server(self, lock_server_log_level=logging.INFO):
        logger = setup_logger("light_scale")
        logger.info("正在启动分布式锁服务器进程...")
        self.start()
        
        logger.info(f"等待服务器就绪信号 (超时: {self.startup_timeout} 秒)...")
        
        # 3. 使用 wait 的 timeout 参数
        is_ready = self.ready_event.wait(timeout=self.startup_timeout)
        
        # 4. 检查 wait 的返回值
        if not is_ready:
            # 如果 is_ready 是 False，说明超时了
            logger.error(f"错误: 服务器在 {self.startup_timeout} 秒内未能启动。正在清理...")
            # 必须手动终止已经启动但未就绪的子进程
            self.terminate()
            self.join()
            # 抛出明确的异常
            raise TimeoutError(f"分布式锁服务器未能在 {self.startup_timeout} 秒内发出就绪信号。")
            
        logger.info("接收到就绪信号，分布式锁服务器已完全启动。")

    def shutdown_lock_server(self):
        logger = setup_logger("light_scale")
        logger.info("[Main] 正在停止服务器进程...")
        if self.is_alive():
            self.terminate()
            self.join()
        logger.info("[Main] 服务器进程已停止。")

class DistributedLock:
    """
    一个分布式锁的客户端，实现了上下文管理器协议。
    """
    def __init__(self, lock_name, host='127.0.0.1', port=9999):
        self.lock_name = lock_name
        self.host = host
        self.port = port
        self._socket = None
        self.logger = setup_logger("light_scale")

    def _connect(self):
        """建立到服务器的连接。"""
        if not self._socket:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))

    def acquire(self):
        """
        请求获取锁，此方法会阻塞直到成功获取锁。
        """
        self._connect()
        # 发送获取锁的请求
        request = f'ACQUIRE {self.lock_name}'
        self._socket.sendall(request.encode('utf-8'))
        
        # 阻塞等待服务器的 "ACQUIRED" 响应
        response = self._socket.recv(1024).decode('utf-8')
        
        if response == 'ACQUIRED':
            self.logger.debug(f"[CLIENT] 成功获取锁: {self.lock_name}")
            return True
        else:
            # 在这个简单实现中，不应该发生这种情况，因为服务器会让我们等待
            self.logger.debug(f"[CLIENT] 获取锁失败: {self.lock_name}, 响应: {response}")
            return False

    def release(self):
        """
        释放锁。
        """
        if not self._socket:
            # 如果没有连接，说明没有持有锁
            return

        try:
            request = f'RELEASE {self.lock_name}'
            self._socket.sendall(request.encode('utf-8'))
            # 等待服务器的确认
            response = self._socket.recv(1024).decode('utf-8')
            if response == 'RELEASED':
                self.logger.debug(f"[CLIENT] 成功释放锁: {self.lock_name}")
            else:
                self.logger.debug(f"[CLIENT] 释放锁时出错: {response}")
        finally:
            # 释放后关闭连接
            self._socket.close()
            self._socket = None

    def __enter__(self):
        """上下文管理器的入口点，获取锁。"""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器的出口点，释放锁。"""
        self.release()

if __name__ == '__main__':
    lock_server = LockServer()
    lock_server.start()