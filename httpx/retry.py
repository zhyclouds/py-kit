"""重试机制装饰器：指数退避 + 随机抖动 + 可定义异常逻辑。"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

__all__ = ["retry_on_failure"]

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def retry_on_failure(
    base_delay: float = 1,
    max_delay: float = 30,
    max_retry: int = 5,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    should_retry: Callable[[Exception], bool] | None = None,
    jitter_ratio: float = 0.5,
) -> Callable:
    """装饰器：失败后按指数退避 + 随机抖动重试，同时支持同步与异步函数。

    Args:
        base_delay: 首次重试的基础延迟（秒）。
        max_delay: 每次重试延迟的硬上限（秒，含抖动）。
        max_retry: 最大尝试次数（含首次调用）。
        retryable_exceptions: 需要重试的异常类型，不在其中的异常直接抛出。
        should_retry: 自定义重试判定函数，入参为捕获的异常，返回 False 则立即抛出；
            提供时优先于 retryable_exceptions，适合按状态码等更细粒度的逻辑。
        jitter_ratio: 抖动系数，取值 [0, 1]，实际延迟为退避值加上
            [0, 退避值 * jitter_ratio] 的随机量，避免多个客户端同时重试造成惊群。

    Returns:
        装饰后的函数，重试耗尽后抛出 RuntimeError（__cause__ 保留最后一次原始异常）。

    用法::

        @retry_on_failure(
            base_delay=1,
            max_delay=30,
            max_retry=5,
            retryable_exceptions=(ConnectionError, TimeoutError),
        )
        def fetch(url: str) -> str: ...

        @retry_on_failure(should_retry=lambda e: getattr(e, "status_code", 0) >= 500)
        async def call_api(path: str) -> dict: ...
    """

    def backoff_delay(attempt: int) -> float:
        """第 attempt 次失败（从 0 开始）后的重试延迟，指数退避后叠加随机抖动。"""
        delay = min(base_delay * (2 ** attempt), max_delay)
        return min(delay + random.uniform(0, delay * jitter_ratio), max_delay)

    def retryable(exc: Exception) -> bool:
        if should_retry is not None:
            return should_retry(exc)
        return isinstance(exc, retryable_exceptions)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(max_retry):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not retryable(e):
                        raise
                    if attempt == max_retry - 1:
                        raise RuntimeError(f"{func.__name__} 在 {max_retry} 次尝试后仍失败") from e
                    delay = backoff_delay(attempt)
                    logger.warning(
                        "%s 第 %d 次失败（%s: %s），%.2fs 后重试",
                        func.__name__, attempt + 1, type(e).__name__, e, delay,
                    )
                    time.sleep(delay)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(max_retry):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if not retryable(e):
                        raise
                    if attempt == max_retry - 1:
                        raise RuntimeError(f"{func.__name__} 在 {max_retry} 次尝试后仍失败") from e
                    delay = backoff_delay(attempt)
                    logger.warning(
                        "%s 第 %d 次失败（%s: %s），%.2fs 后重试",
                        func.__name__, attempt + 1, type(e).__name__, e, delay,
                    )
                    await asyncio.sleep(delay)

        return async_wrapper if inspect.iscoroutinefunction(func) else wrapper

    return decorator
