# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Reference-counted wrapper around :class:`~concurrent.futures.ThreadPoolExecutor`."""

from concurrent import futures


class SharedThreadPoolExecutor(futures.ThreadPoolExecutor):
    """
    Executor that shares one underlying thread pool across multiple owners.

    Call :meth:`share` once per borrower. Each borrower should call
    :meth:`shutdown` exactly once when finished; the wrapped executor is shut
    down only after the last reference is released.

    Notes
    -----
    Intended for single-threaded borrow/release (e.g. main-thread simulator
    lifecycle). No locking is used.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._refs = 0

    def share(self) -> "SharedThreadPoolExecutor":
        """
        Acquire a reference to this shared pool.

        Returns
        -------
        SharedThreadPool
            This instance (for chaining and passing to gRPC / other consumers).
        """
        self._refs += 1
        return self

    @property
    def ref_count(self) -> int:
        """Number of outstanding references acquired via :meth:`share`."""
        return self._refs

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """
        Release one reference to the shared pool.

        The underlying :class:`~concurrent.futures.ThreadPoolExecutor` is shut
        down only when the reference count reaches zero.
        """
        self._refs -= 1
        if self._refs <= 0:
            super().shutdown(wait=wait, cancel_futures=cancel_futures)
