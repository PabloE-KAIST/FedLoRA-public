"""Multi-step invariance tests for the Gate-2 HetLoRA support-restriction primitives.

Proves that the inactive tail rows [rank:] stay EXACTLY invariant across many AdamW steps
(cut -> 0, freeze -> global snapshot), while the active rows [:rank] still train -- and that
a plain gradient mask WITHOUT the clamp/state-clear is NOT sufficient (AdamW momentum drifts
the masked rows), which is the whole reason the extra safeguards exist.

Run: python -m pytest \
     federatedscope/contrib/common/test_hetlora_restriction.py -q
"""
import torch
import torch.nn as nn

from federatedscope.contrib.common.hetlora_restriction import (
    iter_lora_rank_params, capture_active_snapshot, apply_inactive,
    mask_inactive_grads, clear_optimizer_state_inactive,
)


class TinyLoRA(nn.Module):
    """Two LoRA weights with PEFT-like names: '...lora_A.default.weight' (r,in),
    '...lora_B.default.weight' (out,r)."""
    def __init__(self, r=8, d_in=4, d_out=4):
        super().__init__()
        self.lora_A = nn.ModuleDict({'default': nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(r, d_out, bias=False)})

    def forward(self, x):
        return self.lora_B['default'](self.lora_A['default'](x))


def _inactive(model, rank):
    return {n: (p[rank:, :].clone() if a else p[:, rank:].clone())
            for n, p, a in iter_lora_rank_params(model)}


def _active(model, rank):
    return {n: (p[:rank, :].clone() if a else p[:, :rank].clone())
            for n, p, a in iter_lora_rank_params(model)}


def _randomize(model):
    with torch.no_grad():
        for _, p, _ in iter_lora_rank_params(model):
            p.copy_(torch.randn_like(p))


def test_iter_finds_A_and_B():
    m = TinyLoRA(r=8, d_in=4, d_out=6)
    got = {n: (a, tuple(p.shape)) for n, p, a in iter_lora_rank_params(m)}
    assert got['lora_A.default.weight'] == (True, (8, 4))   # (rank, in)
    assert got['lora_B.default.weight'] == (False, (6, 8))  # (out, rank)


def test_apply_inactive_cut_zeros_tail_only():
    m = TinyLoRA(r=8); _randomize(m); rank = 5
    apply_inactive(m, rank, 'cut', {})
    for n, p, a in iter_lora_rank_params(m):
        tail = p[rank:, :] if a else p[:, rank:]
        head = p[:rank, :] if a else p[:, :rank]
        assert torch.count_nonzero(tail) == 0
        assert torch.count_nonzero(head) > 0  # active untouched


def test_apply_inactive_freeze_restores_snapshot():
    m = TinyLoRA(r=8); _randomize(m); rank = 5
    snap = {}
    capture_active_snapshot(m, 8, snap)       # snapshot everything (all active at rank 8)
    tgt = _inactive(m, rank)
    _randomize(m)                             # clobber the model
    apply_inactive(m, rank, 'freeze', snap)   # tail must come back to the snapshot values
    now = _inactive(m, rank)
    for n in tgt:
        assert torch.allclose(now[n], tgt[n])


def _train_step(model, opt, rank, mode, snap, clamp=True, clear=True, mask=True):
    opt.zero_grad()
    model(torch.randn(16, model.lora_A['default'].in_features)).pow(2).mean().backward()
    if mask:
        mask_inactive_grads(model, rank, mode)
    if clear:
        clear_optimizer_state_inactive(opt, model, rank, mode)
    opt.step()
    if clamp:
        apply_inactive(model, rank, mode, snap)


def test_multistep_freeze_invariance():
    torch.manual_seed(0)
    m = TinyLoRA(r=8, d_in=4, d_out=4); _randomize(m); rank = 4
    snap = {}
    capture_active_snapshot(m, 8, snap)          # download-time capture
    apply_inactive(m, rank, 'freeze', snap)      # restore tail
    frozen = _inactive(m, rank); act0 = _active(m, rank)
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    for step in range(15):
        _train_step(m, opt, rank, 'freeze', snap)
        for n, p, a in iter_lora_rank_params(m):
            tail = p[rank:, :] if a else p[:, rank:]
            assert torch.allclose(tail, frozen[n], atol=1e-6), f"freeze drift {n} @step {step}"
    act1 = _active(m, rank)
    assert any(not torch.allclose(act0[n], act1[n]) for n in act0), "active rows must train"


def test_multistep_cut_invariance():
    torch.manual_seed(1)
    m = TinyLoRA(r=8, d_in=4, d_out=4); _randomize(m); rank = 3
    snap = {}
    apply_inactive(m, rank, 'cut', snap)         # zero tail
    act0 = _active(m, rank)
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    for step in range(15):
        _train_step(m, opt, rank, 'cut', snap)
        for n, p, a in iter_lora_rank_params(m):
            tail = p[rank:, :] if a else p[:, rank:]
            assert torch.count_nonzero(tail) == 0, f"cut leak {n} @step {step}"
    act1 = _active(m, rank)
    assert any(not torch.allclose(act0[n], act1[n]) for n in act0), "active rows must train"


def test_gradmask_alone_insufficient_under_adamw():
    """Justifies the clamp + state-clear: train first to build AdamW momentum on the tail,
    then grad-mask WITHOUT clamp/clear -> residual exp_avg drifts the 'frozen' rows."""
    torch.manual_seed(2)
    m = TinyLoRA(r=8, d_in=4, d_out=4); _randomize(m); rank = 4
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    # phase 1: train ALL rows so inactive rows accumulate momentum
    for _ in range(4):
        _train_step(m, opt, rank, 'none', {}, clamp=False, clear=False, mask=False)
    frozen = _inactive(m, rank)
    # phase 2: grad-mask inactive rows but do NOT clear state and do NOT clamp
    drifted = False
    for _ in range(6):
        _train_step(m, opt, rank, 'freeze', {}, clamp=False, clear=False, mask=True)
        now = _inactive(m, rank)
        if any(not torch.allclose(now[n], frozen[n], atol=1e-6) for n in frozen):
            drifted = True
    assert drifted, "AdamW momentum should drift grad-masked rows (clamp+clear are required)"


def test_capture_preserves_inactive_across_shrink():
    """As rank shrinks, a row that goes inactive keeps the global value from its last active
    round (capture only refreshes [:rank])."""
    m = TinyLoRA(r=8, d_in=4, d_out=4); snap = {}
    with torch.no_grad():                       # round A: rank 6, params = 'global A'
        for _, p, _ in iter_lora_rank_params(m):
            p.copy_(torch.full_like(p, 1.0))
    capture_active_snapshot(m, 6, snap)
    gA = _inactive(m, 4)                         # rows [4:6] currently 1.0
    with torch.no_grad():                        # round B: rank 4, new global = 2.0
        for _, p, _ in iter_lora_rank_params(m):
            p.copy_(torch.full_like(p, 2.0))
    capture_active_snapshot(m, 4, snap)          # refresh only [:4]
    apply_inactive(m, 4, 'freeze', snap)         # restore [4:] from snapshot
    now = _inactive(m, 4)
    for n in gA:                                 # rows [4:6] must still be 1.0, not 2.0
        assert torch.allclose(now[n], gA[n]), f"{n} lost its last-active global value"
