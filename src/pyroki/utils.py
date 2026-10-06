# 学习导读：提供墙钟计时和 JIT 内调试日志；日志用 jax.debug.callback 回到宿主执行。
import contextlib
import time
from functools import partial
from typing import Generator

import jax
import termcolor
from loguru import logger


# 在上下文前后记录 time.time 墙钟差；异步 JAX 工作需要调用方另行同步才代表完整数值耗时。
@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    """Context manager for measuring runtime."""
    start_time = time.time()
    print("\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print("========")


# 绑定 loguru 的 function 字段后输出宿主日志，作为 callback 的执行目标。
def _log(fmt: str, *args, **kwargs) -> None:
    logger.bind(function="log").info(fmt, *args, **kwargs)


# 通过 jax.debug.callback 让编译计算期间的信息由宿主日志函数打印。
def jax_log(fmt: str, *args, **kwargs) -> None:
    """Emit a loguru info message from a JITed JAX function."""
    jax.debug.callback(partial(_log, fmt), *args, **kwargs)
