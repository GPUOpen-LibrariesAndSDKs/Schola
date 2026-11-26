# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Unit tests for IREE export of a minimal stateful (LSTM) ScholaModel."""

from __future__ import annotations

from functools import cached_property

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from schola.core.export.iree_export import load_iree_module, save_as_iree
from schola.core.model import ScholaModel, StateMetadata
from schola.core.utils.dict_helpers import NestedDict

pytest.importorskip("iree.turbine")
pytest.importorskip("iree.runtime")


class TinyLSTMScholaModel(ScholaModel):
    """Small LSTM policy used to exercise stateful IREE export."""

    def __init__(self, hidden_size: int = 8):
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        super().__init__(obs_space, action_space)
        self.hidden_size = hidden_size
        self.lstm = th.nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True)
        self.head = th.nn.Linear(hidden_size, 2)

    @cached_property
    def initial_state_dict(self) -> NestedDict[str, th.Tensor]:
        # Shape without batch: [num_layers, hidden]
        return {
            "h": th.zeros(1, self.hidden_size),
            "c": th.zeros(1, self.hidden_size),
        }

    @cached_property
    def state_metadata(self) -> NestedDict[str, StateMetadata]:
        meta = StateMetadata(has_seq_dim=False)
        return {"h": meta, "c": meta}

    def forward(self, obs: th.Tensor, state_h: th.Tensor, state_c: th.Tensor):
        # obs: [B, 4] -> [B, 1, 4]; state: [B, 1, H] -> [1, B, H]
        x = obs.unsqueeze(1)
        h0 = state_h.transpose(0, 1).contiguous()
        c0 = state_c.transpose(0, 1).contiguous()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        action = self.head(out[:, -1, :])
        # Return state as [B, 1, H] to match Schola naming conventions
        return action, hn.transpose(0, 1).contiguous(), cn.transpose(0, 1).contiguous()


def test_iree_export_lstm_matches_pytorch(tmp_path):
    model = TinyLSTMScholaModel()
    model.eval()
    vmfb_path = tmp_path / "lstm.vmfb"
    save_as_iree(model, vmfb_path)

    assert vmfb_path.exists()
    _, metadata = load_iree_module(vmfb_path)
    assert {s.in_name for s in metadata.state} == {"state_in_h", "state_in_c"}
    assert {s.out_name for s in metadata.state} == {"state_out_h", "state_out_c"}

    obs = th.as_tensor(np.random.randn(1, 4).astype(np.float32))
    h = th.zeros(1, 1, model.hidden_size)
    c = th.zeros(1, 1, model.hidden_size)
    with th.no_grad():
        pt_action, pt_h, pt_c = model(obs, h, c)

    vm_module, meta = load_iree_module(vmfb_path)
    entry = getattr(vm_module, meta.entry_function)
    outputs = entry(obs.numpy(), h.numpy(), c.numpy())
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    host = [o.to_host() if hasattr(o, "to_host") else np.asarray(o) for o in outputs]

    np.testing.assert_allclose(host[0], pt_action.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(host[1], pt_h.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(host[2], pt_c.numpy(), rtol=1e-4, atol=1e-4)
