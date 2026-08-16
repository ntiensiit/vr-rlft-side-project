"""Phase 4 flow training tests."""

from __future__ import annotations

import pytest
import torch

from grasping_ai.models.flow import (
    FlowFieldNet,
    FlowGeneratorModel,
    load_flow_model_from_state,
)


def test_flow_model_forward_delegates_to_flow_field() -> None:
    """Verify ``FlowGeneratorModel.forward`` returns flow-field predictions.

    Returns:
        None. Asserts output shape matches the input grasp batch.
    """
    model = FlowGeneratorModel(feature_dim=8, hidden_dim=8, num_layers=1)
    x = torch.zeros(2, 9)
    cond = torch.zeros(2, 8)
    out = model.forward(x, cond)
    if not (out.shape == (2, 9)):
        raise AssertionError


def test_flow_network_builder_and_sampler_additional_coverage() -> None:
    """Verify flow network setup errors when using invalid state dict mapping structures."""
    net = FlowFieldNet(8, 16, 2)
    if not (isinstance(net, FlowFieldNet)):
        raise AssertionError  # noqa: TRY004  # value expectation, not a signature type check

    with pytest.raises(TypeError, match=r"checkpoint\['model_state_dict'\] must be a dictionary"):
        load_flow_model_from_state({"model_state_dict": "not_a_dict"}, 8, 16, 2, "cpu")
