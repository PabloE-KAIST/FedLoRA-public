"""HetLoRA Gate-2 support-restriction primitives (fixed rank-200 adapter, no resize).

Under the corrected mechanism the physical LoRA adapter is ALWAYS max_rank (200) rows for
every client; a client's "rank" R is only its FEDERATED support. Reducing R to a matched
final support R_k* restricts the eval-relevant capacity (eval loads the downloaded global,
tail zeroed). These primitives realize the two matched support-restriction operations on the
INACTIVE tail rows [R:200] of every LoRA A (dim 0) / B (dim 1):

  * mode='cut'    -> inactive rows are ZEROED (removed from the forward pass); irreversible.
  * mode='freeze' -> inactive rows are held at the LAST SERVER-AGGREGATED GLOBAL value
                     (captured from the download, NOT the client's locally-trained tail),
                     kept in the forward pass, excluded from training/upload/aggregation.

AdamW safety (critical): a plain gradient mask is NOT sufficient -- AdamW keeps momentum
(exp_avg) and variance (exp_avg_sq) state, so a row that was trained before it became
inactive would keep drifting at zero grad. So we (a) clear optimizer state for inactive rows
at restriction time, and (b) HARD-CLAMP the inactive rows after EVERY optimizer step (cut->0,
freeze->snapshot). The post-step clamp is the invariance guarantee; the grad mask + state
clear avoid the optimizer wastefully fighting it. See test_hetlora_restriction.py.

All functions are pure (operate on a model / optimizer / snapshot dict) so they are unit
tested without the full FL stack.
"""
import torch


def iter_lora_rank_params(model):
    """Yield (name, param, is_A) for LoRA A/B weight params (rank = dim0 for A, dim1 for B)."""
    for name, p in model.named_parameters():
        if not name.endswith('weight'):
            continue
        if 'lora_A' in name:
            yield name, p, True
        elif 'lora_B' in name:
            yield name, p, False


def capture_active_snapshot(model, rank, snapshot):
    """Refresh snapshot's ACTIVE rows [:rank] from the current params.

    Called right AFTER the global download (params[:rank] == server-aggregated global,
    pre-training) and BEFORE local training. Inactive rows [rank:] are left untouched, so
    once a row goes inactive its snapshot holds the global value it had the last round it was
    federated. On first call the snapshot is initialized to a full clone.
    """
    with torch.no_grad():
        for name, p, is_A in iter_lora_rank_params(model):
            full = snapshot.get(name)
            if full is None:
                full = p.detach().clone()
                snapshot[name] = full
            if is_A:
                full[:rank, :] = p.data[:rank, :]
            else:
                full[:, :rank] = p.data[:, :rank]


def apply_inactive(model, rank, mode, snapshot):
    """Set inactive tail rows [rank:] to their target: cut->0, freeze->snapshot.

    Used both after the download (restore the tail the download just zeroed) and after every
    optimizer step (the hard invariance clamp). No-op for mode not in {cut, freeze}.
    """
    if mode not in ('cut', 'freeze'):
        return
    with torch.no_grad():
        for name, p, is_A in iter_lora_rank_params(model):
            if is_A:
                if mode == 'cut':
                    p.data[rank:, :] = 0.0
                elif name in snapshot:
                    p.data[rank:, :] = snapshot[name][rank:, :].to(p.device, p.dtype)
            else:
                if mode == 'cut':
                    p.data[:, rank:] = 0.0
                elif name in snapshot:
                    p.data[:, rank:] = snapshot[name][:, rank:].to(p.device, p.dtype)


def mask_inactive_grads(model, rank, mode):
    """Zero grads on inactive tail rows [rank:] before the optimizer step."""
    if mode not in ('cut', 'freeze'):
        return
    for name, p, is_A in iter_lora_rank_params(model):
        if p.grad is None:
            continue
        if is_A:
            p.grad[rank:, :] = 0.0
        else:
            p.grad[:, rank:] = 0.0


def clear_optimizer_state_inactive(optimizer, model, rank, mode):
    """Zero AdamW moment state (exp_avg/exp_avg_sq) for inactive rows so no momentum leaks."""
    if mode not in ('cut', 'freeze') or optimizer is None:
        return
    for name, p, is_A in iter_lora_rank_params(model):
        st = optimizer.state.get(p)
        if not st:
            continue
        for k in ('exp_avg', 'exp_avg_sq'):
            v = st.get(k)
            if torch.is_tensor(v):
                if is_A:
                    v[rank:, :] = 0.0
                else:
                    v[:, rank:] = 0.0
