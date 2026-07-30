# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for SharedThreadPool reference counting."""

import time
from concurrent import futures

from schola.core.utils.shared_thread_pool_executor import SharedThreadPoolExecutor


def test_shutdown_only_after_last_reference_released():
    pool = SharedThreadPoolExecutor(max_workers=2)
    pool.share()
    pool.share()

    assert pool.ref_count == 2

    pool.shutdown(wait=False)
    assert pool.ref_count == 1

    future = pool.submit(time.sleep, 0)
    future.result(timeout=1)

    pool.shutdown(wait=True)
    assert pool.ref_count == 0


def test_submit_delegates_to_underlying_executor():
    pool = SharedThreadPoolExecutor(max_workers=1).share()

    future = pool.submit(lambda x: x + 1, 41)
    assert future.result(timeout=1) == 42

    pool.shutdown(wait=True)


def test_shutdown_without_share_is_noop():
    pool = SharedThreadPoolExecutor(max_workers=1)
    pool.shutdown(wait=True)
    assert pool.ref_count == 0
