# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for ``GrpcProtocolConfig`` multi-simulator protocol construction."""

from schola.scripts.common.settings import GrpcProtocolConfig, PortOffsetMode


def test_fixed_port_mode_reuses_base_port_on_every_worker():
    protocols = GrpcProtocolConfig(
        port=9001,
        port_offset_mode=PortOffsetMode.FIXED,
        grpc_close_timeout=15,
    ).make_n_async(3)
    assert [p.port for p in protocols] == [9001, 9001, 9001]


def test_make_n_async_forwards_grpc_close_timeout():
    protocols = GrpcProtocolConfig(
        port=9001,
        port_offset_mode=PortOffsetMode.FIXED,
        grpc_close_timeout=15,
    ).make_n_async(3)
    assert all(getattr(p, "grpc_close_timeout", None) == 15 for p in protocols)
