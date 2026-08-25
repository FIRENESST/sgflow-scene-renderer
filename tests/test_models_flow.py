import pytest
import torch

from sgflow.config import SGFlowConfig
from sgflow.cuda_ops import ssm_scan
from sgflow.flow_matching import RectifiedFlow
from sgflow.models import LatentBottleneck, SceneDenoiser, timestep_embedding


def tiny_config():
    return SGFlowConfig(
        d_model=8, n_layers=1, n_latents=2, d_state=3, expand=1,
        max_objects=4, d_appearance=1, ssm_chunk=2, flow_steps=3,
    )


def test_ssm_scan_has_input_and_parameter_gradients():
    torch.manual_seed(0)
    u = torch.randn(2, 5, 3, requires_grad=True)
    delta = torch.rand(2, 5, 3, requires_grad=True)
    A = -torch.rand(3, 2)
    A.requires_grad_()
    bm = torch.randn(2, 5, 2, requires_grad=True)
    cm = torch.randn(2, 5, 2, requires_grad=True)
    y = ssm_scan(u, delta, A, bm, cm, chunk=2)
    y.square().mean().backward()
    for value in (u, delta, A, bm, cm):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    with pytest.raises(ValueError, match="positive"):
        ssm_scan(u.detach(), delta.detach(), A.detach(), bm.detach(), cm.detach(), chunk=0)


def test_queries_are_live_and_padded_outputs_are_zero():
    cfg = tiny_config()
    model = SceneDenoiser(cfg)
    z = torch.randn(2, cfg.max_objects, cfg.latent_dim)
    cat = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
    text = torch.randn(2, 3, cfg.d_model)
    text_mask = torch.tensor([[True, True, False], [True, True, True]])
    obj_mask = cat.ne(0)
    out = model(z, torch.tensor([0.2, 0.7]), cat, text, text_mask, obj_mask)
    out.sum().backward()
    block = model.blocks[0].bottleneck
    assert block.q_in.weight.grad is not None and block.q_in.weight.grad.abs().sum() > 0
    assert block.q_rd.weight.grad is not None and block.q_rd.weight.grad.abs().sum() > 0
    assert torch.equal(out[~obj_mask], torch.zeros_like(out[~obj_mask]))


def test_timestep_and_attention_validation():
    assert timestep_embedding(torch.tensor([0.5]), 7).shape == (1, 7)
    assert timestep_embedding(torch.tensor([0.5]), 1).shape == (1, 1)
    with pytest.raises(ValueError, match="divisible"):
        LatentBottleneck(10, 2, n_heads=8)


def test_sparse_structure_loss_backpropagates_and_sampling_is_deterministic():
    cfg = tiny_config()
    flow = RectifiedFlow(cfg)
    model = SceneDenoiser(cfg)
    cat = torch.tensor([[1, 0, 0, 0], [2, 0, 0, 0]])
    obj_mask = cat.ne(0)
    z1 = torch.randn(2, cfg.max_objects, cfg.latent_dim)
    text = torch.randn(2, 3, cfg.d_model)
    text_mask = torch.ones(2, 3, dtype=torch.bool)
    loss, metrics = flow.loss(model, z1, cat, text, text_mask, obj_mask)
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in metrics.values())
    loss.backward()
    assert model.struct.ff[-1].weight.grad is not None

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    sample1 = flow.sample(model, cat, text, text_mask, obj_mask, steps=2, generator=g1)
    sample2 = flow.sample(model, cat, text, text_mask, obj_mask, steps=2, generator=g2)
    assert torch.equal(sample1, sample2)
    assert torch.equal(sample1[~obj_mask], torch.zeros_like(sample1[~obj_mask]))
    with pytest.raises(ValueError, match="positive"):
        flow.sample(model, cat, text, text_mask, obj_mask, steps=0)
