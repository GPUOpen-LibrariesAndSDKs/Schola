# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import builtins

from schola.scripts.common.console import maybe_tqdm


def test_maybe_tqdm_yields_identity_when_disabled() -> None:
    with maybe_tqdm(False) as tqdm_fn:
        assert list(tqdm_fn(range(3))) == [0, 1, 2]


def test_maybe_tqdm_yields_wrapper_when_enabled() -> None:
    with maybe_tqdm(True) as tqdm_fn:
        wrapped = tqdm_fn(range(2), desc="Fabrica")
        assert list(wrapped) == [0, 1]


def test_maybe_tqdm_falls_back_when_tqdm_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm":
            raise ModuleNotFoundError("simulated missing tqdm")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    with maybe_tqdm(True) as tqdm_fn:
        assert list(tqdm_fn(range(2))) == [0, 1]
