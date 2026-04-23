import logging
from logging.handlers import QueueHandler, QueueListener
from typing import Optional

import torch.distributed as dist
from megatron.core import mpu
import multiprocessing as mp


_LOGGING_QUEUE: Optional[mp.Queue] = None
_LOGGING_LISTENER: Optional[QueueListener] = None
_QUEUE_HANDLER_FLAG = "_light_scale_queue_handler"


class _FormattedQueueHandler(QueueHandler):
    """QueueHandler that formats log records before enqueue."""

    def __init__(self, queue: mp.Queue, formatter: logging.Formatter):
        super().__init__(queue)
        self._formatter = formatter

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        record.msg = self._formatter.format(record)
        record.args = None
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return super().prepare(record)


class _PlainStreamHandler(logging.StreamHandler):
    """Stream handler that trusts record.msg has been formatted."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        message = record.msg
        if message is None:
            return ""
        return message if isinstance(message, str) else str(message)

class DistributedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tp_rank=0, pp_rank=0, cp_rank=0, dp_rank=0):
        super().__init__(fmt, datefmt)
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.cp_rank = cp_rank
        self.dp_rank = dp_rank

    def format(self, record):
        # 给 record 动态添加 tp_rank 和 pp_rank
        record.tp = self.tp_rank
        record.pp = self.pp_rank
        record.cp = self.cp_rank
        record.dp = self.dp_rank
        return super().format(record)
    
def _build_formatter(setup_distributed: bool) -> DistributedFormatter:
    if setup_distributed:
        assert dist.is_initialized() and mpu.is_initialized()
        fmt = '[%(asctime)s] [%(process)d] [TP%(tp)d CP%(cp)d PP%(pp)d DP%(dp)d] [%(levelname)s] %(message)s'
        return DistributedFormatter(
            fmt,
            datefmt='%Y/%m/%d %H:%M:%S',
            tp_rank=mpu.get_tensor_model_parallel_rank(),
            pp_rank=mpu.get_pipeline_model_parallel_rank(),
            cp_rank=mpu.get_context_parallel_rank(),
            dp_rank=mpu.get_data_parallel_rank(),
        )
    fmt = '[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s'
    return DistributedFormatter(
        fmt,
        datefmt='%Y/%m/%d %H:%M:%S',
    )

def setup_logger(name: str = None, setup_distributed: bool = True, level=logging.INFO):
    if name is None:
        name = __name__
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    if setup_distributed:
        assert dist.is_initialized() and mpu.is_initialized()
        fmt = '[%(asctime)s] [%(process)d] [TP%(tp)d CP%(cp)d PP%(pp)d DP%(dp)d] [%(levelname)s] %(message)s'
        formatter = DistributedFormatter(
            fmt,
            datefmt='%Y/%m/%d %H:%M:%S',
            tp_rank=mpu.get_tensor_model_parallel_rank(),
            pp_rank=mpu.get_pipeline_model_parallel_rank(),
            cp_rank=mpu.get_context_parallel_rank(),
            dp_rank=mpu.get_data_parallel_rank()
        )
    else:
        fmt = '[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s'
        formatter = DistributedFormatter(
            fmt,
            datefmt='%Y/%m/%d %H:%M:%S'
        )
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

def setup_logger_v2_main_process(name: str = None, setup_distributed: bool = True, level=logging.INFO):
    global _LOGGING_QUEUE, _LOGGING_LISTENER
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    formatter = _build_formatter(setup_distributed)
    if getattr(logger, _QUEUE_HANDLER_FLAG, False):
        return logger
    if _LOGGING_QUEUE is None:
        _LOGGING_QUEUE = mp.Queue()
    queue_handler = _FormattedQueueHandler(_LOGGING_QUEUE, formatter)
    queue_handler.setLevel(level)
    logger.handlers.clear()
    logger.handlers.append(queue_handler)
    logger.setLevel(level)
    logger.propagate = False
    setattr(logger, _QUEUE_HANDLER_FLAG, True)
    if _LOGGING_LISTENER is None:
        plain_handler = _PlainStreamHandler()
        plain_handler.setLevel(level)
        listener = QueueListener(
            _LOGGING_QUEUE,
            plain_handler,
            respect_handler_level=True,
        )
        listener.start()
        _LOGGING_LISTENER = listener
    return logger

def setup_logger_v2_sub_process(name: str = None, setup_distributed: bool = True, level=logging.INFO, log_queue: mp.Queue = None):
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    # if name in logging.Logger.manager.loggerDict:
    #     return logging.getLogger(name)
    if getattr(logger, _QUEUE_HANDLER_FLAG, False):
        return logger
    global _LOGGING_QUEUE
    if log_queue is None:
        raise ValueError("log_queue must be provided for sub processes")
    _LOGGING_QUEUE = log_queue
    formatter = _build_formatter(setup_distributed)
    fmt = '[%(asctime)s] [%(process)d] [async_rollout] [%(levelname)s] %(message)s'
    fmt = fmt.replace("async_rollout", name)
    formatter = logging.Formatter(
        fmt,
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    queue_handler = _FormattedQueueHandler(log_queue, formatter)
    queue_handler.setLevel(level)
    logger.handlers.clear()
    logger.handlers.append(queue_handler)
    logger.setLevel(level)
    logger.propagate = False
    setattr(logger, _QUEUE_HANDLER_FLAG, True)
    return logger

def get_logging_queue() -> mp.Queue:
    if _LOGGING_QUEUE is None:
        raise RuntimeError(
            "Logging queue is not initialized. Call setup_logger_v2_main_process() "
            "in the main process or setup_logger_v2_sub_process() with a provided queue before accessing it."
        )
    return _LOGGING_QUEUE