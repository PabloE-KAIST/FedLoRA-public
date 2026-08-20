#!/usr/bin/env python3
"""
Memory profiling for LoRA rank budget determination.

Runs a short training pass (30 steps) at each specified rank and reports peak
GPU memory.  Designed to run inside a fedlora-worker container on a target
device, or directly on a host with a GPU.

Usage (inside container or host):
    python 1_scripts/distributed/log_tools/profile_memory.py \
        --config 2_yamls/fedit/fedit_distributed.yaml \
        --ranks 8 32 64 100 150 200 \
        --steps 30

Output: a table of rank → peak_memory_MB → status (ok / OOM).
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = None
if not hasattr(torch, "float8_e5m2"):
    torch.float8_e5m2 = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def profile_rank(cfg_template, rank, num_steps, batch_size, nbits=16):
    """Run training for `num_steps` at the given LoRA rank. Return peak MB."""
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.auxiliaries.model_builder import get_model

    cfg = global_cfg.clone()
    cfg.merge_from_file(cfg_template)

    cfg.defrost()
    cfg.federate.mode = "standalone"
    cfg.computation_quantization.method = "none"
    cfg.computation_quantization.nbits = nbits

    worker_prefix = "/workspace/models/"
    if cfg.model.type.startswith(worker_prefix):
        host_model_root = os.environ.get(
            "FEDLORA_HOST_MODEL_ROOT", os.path.expanduser("~/models")
        )
        if os.path.isdir(host_model_root):
            cfg.model.type = cfg.model.type.replace(
                worker_prefix, host_model_root + "/", 1
            )

    for adapter_root in (cfg.glue.adapter, cfg.llm.adapter):
        if hasattr(adapter_root, "args") and adapter_root.args:
            a = adapter_root.args[0]
            if isinstance(a, dict):
                a["r"] = rank
            else:
                a.r = rank
        adapter_root.max_rank = rank

    cfg.train.local_update_steps = num_steps
    cfg.train.batch_or_epoch = "batch"
    cfg.dataloader.batch_size = batch_size
    cfg.eval.count_flops = False
    cfg.outdir = f"/tmp/profile_rank_{rank}"
    os.makedirs(cfg.outdir, exist_ok=True)
    cfg.freeze()

    model = get_model(cfg, local_data=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "rank=%d  trainable=%s / %s params",
        rank,
        f"{trainable:,}",
        f"{total:,}",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4,
        weight_decay=0.01,
    )

    seq_len = cfg.glue.max_length if hasattr(cfg, "glue") else 128
    vocab_size = model.config.vocab_size if hasattr(model, "config") else 50265

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size, seq_len, vocab_size):
            self.size = size
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
                "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                "labels": torch.tensor(idx % 2, dtype=torch.long),
            }

    loader = torch.utils.data.DataLoader(
        DummyDataset(num_steps * batch_size * 2, seq_len, vocab_size),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device) / 1e6

    model.train()
    step = 0
    t0 = time.time()
    for batch in loader:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        if step % 10 == 0:
            torch.cuda.synchronize(device)
            cur_peak = torch.cuda.max_memory_allocated(device) / 1e6
            logger.info(
                "  step %d/%d  loss=%.4f  peak=%.0f MB",
                step, num_steps, loss.item(), cur_peak,
            )

    torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    reserved_mb = torch.cuda.memory_reserved(device) / 1e6

    del model, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "rank": rank,
        "peak_allocated_mb": round(peak_mb, 1),
        "peak_reserved_mb": round(reserved_mb, 1),
        "mem_before_mb": round(mem_before, 1),
        "trainable_params": trainable,
        "steps": step,
        "wall_s": round(elapsed, 1),
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA rank memory profiler")
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument(
        "--ranks", type=int, nargs="+", required=True,
        help="LoRA ranks to sweep (ascending recommended)",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=16, choices=[16, 32],
                        help="16 = auto bf16/fp32 per GPU; 32 = force fp32")
    parser.add_argument("--output", default="", help="JSON output path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available — cannot profile GPU memory")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e6
    logger.info("GPU: %s  total_memory=%.0f MB", gpu_name, gpu_mem_total)

    results = []
    for rank in sorted(args.ranks):
        logger.info("=" * 60)
        logger.info("Profiling rank=%d ...", rank)
        try:
            r = profile_rank(args.config, rank, args.steps, args.batch_size,
                            nbits=args.nbits)
            results.append(r)
            logger.info(
                "rank=%d  peak=%.0f MB (%.0f%% of %.0f MB)  wall=%.1fs",
                rank, r["peak_allocated_mb"],
                100 * r["peak_allocated_mb"] / gpu_mem_total,
                gpu_mem_total, r["wall_s"],
            )
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            results.append({
                "rank": rank,
                "peak_allocated_mb": None,
                "status": "OOM",
            })
            logger.warning("rank=%d  OOM — skipping higher ranks", rank)
            break
        except Exception as e:
            results.append({
                "rank": rank,
                "peak_allocated_mb": None,
                "status": f"error: {e}",
            })
            logger.exception("rank=%d failed", rank)

    print("\n" + "=" * 70)
    print(f"{'Rank':>6}  {'Peak MB':>10}  {'% GPU':>7}  {'Params':>12}  "
          f"{'Wall(s)':>8}  {'Status'}")
    print("-" * 70)
    for r in results:
        peak = r.get("peak_allocated_mb")
        pct = f"{100 * peak / gpu_mem_total:.1f}%" if peak else "—"
        params = f"{r.get('trainable_params', 0):,}" if r.get("trainable_params") else "—"
        wall = f"{r.get('wall_s', 0):.1f}" if r.get("wall_s") else "—"
        print(f"{r['rank']:>6}  {peak or '—':>10}  {pct:>7}  {params:>12}  "
              f"{wall:>8}  {r['status']}")
    print("=" * 70)

    if args.output:
        summary = {
            "gpu": gpu_name,
            "gpu_total_mb": round(gpu_mem_total, 0),
            "batch_size": args.batch_size,
            "nbits": args.nbits,
            "steps": args.steps,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
