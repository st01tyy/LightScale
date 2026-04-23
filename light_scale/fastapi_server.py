from fastapi import FastAPI, Request
from multiprocessing import Process, Queue
import uvicorn
import time
from light_scale.logger_utils import setup_logger

class FastAPIServer:
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        self.process = None
        self.queue = Queue()
        
    def run(self, queue):
        app = FastAPI()
        logger = setup_logger("fastapi_server", setup_distributed=False)

        @app.get('/')
        async def home():
            return {"message": "FastAPI server is running!"}

        @app.post('/call')
        async def call(request: Request):
            logger.info("triggered")
            data = await request.json()
            logger.info(data)
            queue.put(data)
            logger.info("before await queue.get()")
            response = queue.get()  # 等待子进程处理并返回消息
            return {"response": response}

        uvicorn.run(app, host=self.host, port=self.port)
    
    def start(self):
        if self.process is None or not self.process.is_alive():
            self.process = Process(target=self.run, args=(self.queue,))
            self.process.start()
            time.sleep(1)  # 等待服务器启动
            print(f"FastAPI server started on {self.host}:{self.port}")
    
    def stop(self):
        if self.process and self.process.is_alive():
            # self.process.terminate()
            self.process.kill()
            self.process.join()
            print("FastAPI server stopped.")
    
    def send_message(self, message):
        self.queue.put(message)
        print(f"Message sent: {message}")
    
    def read_message(self):
        return self.queue.get()

if __name__ == "__main__":
    server = FastAPIServer()
    server.start()
    
    try:
        while True:
            msg = input("Enter a message to send to the FastAPI server (or 'exit' to stop): ")
            if msg.lower() == 'exit':
                break
            server.send_message(msg)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
