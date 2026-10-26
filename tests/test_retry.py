"""retry_on_failure 装饰器的单元测试。

运行方式::

    python -m unittest discover -s tests -v
"""

import inspect
import unittest
from unittest import mock

from httpx.retry import retry_on_failure


def make_flaky(fail_times: int, exc: Exception | None = None):
    """返回 (函数, 调用记录列表)：前 fail_times 次调用抛 exc，之后返回 "ok"。"""
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) <= fail_times:
            raise exc or ConnectionError("boom")
        return "ok"

    return flaky, calls


def make_async_flaky(fail_times: int, exc: Exception | None = None):
    """make_flaky 的异步版本。"""
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) <= fail_times:
            raise exc or ConnectionError("boom")
        return "ok"

    return flaky, calls


def no_jitter():
    """固定 random.uniform 返回 0，让退避序列可精确断言。"""
    return mock.patch("httpx.retry.random.uniform", return_value=0.0)


def max_jitter():
    """固定 random.uniform 返回区间上界，即最大抖动。"""
    return mock.patch("httpx.retry.random.uniform", side_effect=lambda lo, hi: hi)


class SyncRetryTest(unittest.TestCase):
    """同步函数的重试行为。"""

    def test_first_call_success_no_retry(self):
        flaky, calls = make_flaky(0)
        retry = retry_on_failure()(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep:
            result = retry()
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_retry_until_success(self):
        flaky, calls = make_flaky(2)
        retry = retry_on_failure(base_delay=1, max_delay=30, max_retry=5)(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep, no_jitter():
            result = retry()
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 3)
        # 无抖动时的指数退避序列
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])

    def test_exhaust_retries_raises_runtime_error_with_cause(self):
        flaky, calls = make_flaky(1000)
        retry = retry_on_failure(base_delay=1, max_retry=3)(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep, no_jitter():
            with self.assertRaises(RuntimeError) as ctx:
                retry()
        self.assertEqual(len(calls), 3)  # max_retry 为总尝试次数（含首次）
        self.assertIsInstance(ctx.exception.__cause__, ConnectionError)
        # 最后一次失败后不再等待
        self.assertEqual(sleep.call_count, 2)

    def test_non_retryable_exception_propagates_immediately(self):
        flaky, calls = make_flaky(1000, exc=ValueError("bad input"))
        retry = retry_on_failure(retryable_exceptions=(ConnectionError,))(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep:
            with self.assertRaises(ValueError):
                retry()
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_max_delay_caps_backoff(self):
        flaky, _ = make_flaky(4)
        retry = retry_on_failure(base_delay=10, max_delay=25, max_retry=5)(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep, no_jitter():
            retry()
        # 失败 4 次、第 5 次成功，退避序列 10, 20, 40, 80 -> 40 与 80 均封顶为 25
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [10, 20, 25, 25])

    def test_jitter_added_on_top_of_backoff(self):
        flaky, _ = make_flaky(1)
        retry = retry_on_failure(base_delay=2, max_delay=100, jitter_ratio=0.5)(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep, max_jitter():
            retry()
        # 退避 2 + 最大抖动 2 * 0.5 = 3
        self.assertEqual(sleep.call_args, mock.call(3.0))

    def test_jitter_capped_by_max_delay(self):
        flaky, _ = make_flaky(1)
        retry = retry_on_failure(base_delay=10, max_delay=11, jitter_ratio=0.5)(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep, max_jitter():
            retry()
        # 退避 10 + 抖动 5 = 15，但硬上限为 11
        self.assertEqual(sleep.call_args, mock.call(11))

    def test_should_retry_false_raises_immediately(self):
        flaky, calls = make_flaky(1000)
        retry = retry_on_failure(should_retry=lambda e: False)(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep:
            with self.assertRaises(ConnectionError):
                retry()
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_should_retry_overrides_retryable_exceptions(self):
        # ValueError 在异常元组中，但 should_retry 判定不重试
        flaky, calls = make_flaky(1000, exc=ValueError("no retry"))
        retry = retry_on_failure(
            retryable_exceptions=(ValueError,),
            should_retry=lambda e: False,
        )(flaky)
        with mock.patch("httpx.retry.time.sleep") as sleep:
            with self.assertRaises(ValueError):
                retry()
        self.assertEqual(len(calls), 1)

    def test_should_retry_receives_exception(self):
        received = []

        def record(exc):
            received.append(exc)
            return True

        flaky, _ = make_flaky(1)
        retry = retry_on_failure(should_retry=record)(flaky)
        with mock.patch("httpx.retry.time.sleep"), no_jitter():
            retry()
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], ConnectionError)
        self.assertEqual(str(received[0]), "boom")

    def test_should_retry_custom_logic_by_status_code(self):
        class APIError(Exception):
            def __init__(self, status_code: int):
                super().__init__(f"HTTP {status_code}")
                self.status_code = status_code

        errors = iter([APIError(500), APIError(503), APIError(404)])
        calls = []

        def api_call():
            calls.append(1)
            raise next(errors)

        retry = retry_on_failure(should_retry=lambda e: e.status_code >= 500)(api_call)
        with mock.patch("httpx.retry.time.sleep") as sleep:
            with self.assertRaises(APIError) as ctx:
                retry()
        # 500、503 重试，404 直接抛出
        self.assertEqual(len(calls), 3)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(sleep.call_count, 2)

    def test_preserves_function_metadata(self):
        @retry_on_failure()
        def fetch_data():
            """获取数据。"""

        self.assertEqual(fetch_data.__name__, "fetch_data")
        self.assertEqual(fetch_data.__doc__, "获取数据。")

    def test_arguments_passed_through(self):
        recorded = {}

        @retry_on_failure()
        def compute(a, b, *, factor=1):
            recorded.update(a=a, b=b, factor=factor)
            return (a + b) * factor

        self.assertEqual(compute(2, 3, factor=10), 50)
        self.assertEqual(recorded, {"a": 2, "b": 3, "factor": 10})

    def test_logs_warning_on_retry(self):
        flaky, _ = make_flaky(1)
        retry = retry_on_failure(base_delay=1)(flaky)
        with mock.patch("httpx.retry.time.sleep"), no_jitter():
            with self.assertLogs("httpx.retry", level="WARNING") as logs:
                retry()
        self.assertIn("flaky 第 1 次失败", logs.output[0])
        self.assertIn("ConnectionError", logs.output[0])


class AsyncRetryTest(unittest.IsolatedAsyncioTestCase):
    """异步函数的重试行为。"""

    async def test_async_retry_until_success(self):
        flaky, calls = make_async_flaky(2)
        retry = retry_on_failure(base_delay=1, max_retry=5)(flaky)
        with (
            mock.patch("httpx.retry.asyncio.sleep", new_callable=mock.AsyncMock) as sleep,
            no_jitter(),
        ):
            result = await retry()
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])

    async def test_async_exhaust_retries(self):
        flaky, calls = make_async_flaky(1000)
        retry = retry_on_failure(max_retry=3)(flaky)
        with mock.patch("httpx.retry.asyncio.sleep", new_callable=mock.AsyncMock) as sleep:
            with self.assertRaises(RuntimeError) as ctx:
                await retry()
        self.assertEqual(len(calls), 3)
        self.assertIsInstance(ctx.exception.__cause__, ConnectionError)
        self.assertEqual(sleep.call_count, 2)

    async def test_async_non_retryable_propagates_immediately(self):
        flaky, calls = make_async_flaky(1000, exc=ValueError("bad"))
        retry = retry_on_failure(retryable_exceptions=(TimeoutError,))(flaky)
        with mock.patch("httpx.retry.asyncio.sleep", new_callable=mock.AsyncMock) as sleep:
            with self.assertRaises(ValueError):
                await retry()
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_wraps_sync_and_async_differently(self):
        @retry_on_failure()
        async def call_api(): ...

        @retry_on_failure()
        def sync_call(): ...

        self.assertTrue(inspect.iscoroutinefunction(call_api))
        self.assertFalse(inspect.iscoroutinefunction(sync_call))


if __name__ == "__main__":
    unittest.main()
