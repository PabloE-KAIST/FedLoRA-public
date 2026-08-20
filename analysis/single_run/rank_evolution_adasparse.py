#!/usr/bin/env python3
"""
AdaSparse-LoRA Rank Evolution Visualization (single-run logs)

Parses a FederatedScope exp_print.log generated with AdaSparse-LoRA and
visualizes how per-client logical LoRA ranks evolve over training rounds.

This version:
  - plots cumulative savings in a separate subplot on the right
  - uses a dedicated middle panel for the legend and stats (less wasted space)
  - shows each client's initial rank in the legend
  - computes savings against each client's own initial rank, not a single
    global baseline rank

That client-specific baseline point is important for heterogeneous runs:
different initial ranks (8, 30, 64, 200, ...) are not themselves "savings".
Savings should only count rank decreases relative to each client's own
starting rank.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


class TeeStdout:
    """Duplicate console output to both terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def prepare_run_output_dir(input_path, subdir_name):
    """Create a sibling subfolder next to the input file for all generated outputs."""
    output_dir = Path(input_path).resolve().parent / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


ROUND_START_RE = re.compile(
    r"Starting (?:a new training round|training) \(Round #(?P<round>\d+)\)"
)

# Server init line in current logs
ADASPARSE_INIT_RE = re.compile(
    r"Initialized client indices: "
    r"n_clients=(?P<n>\d+), "
    r"(?:default_init_rank|configured_init_rank|init_rank)=(?P<init>\d+), "
    r"rank_min=(?P<min>\d+), rank_max=(?P<max>\d+)"
)

# Current client init line
CLIENT_ENABLE_RE = re.compile(
    r"Client (?P<cid>\d+): enabled, "
    r"configured_init_rank=(?P<init>\d+), .*?init_source=(?P<source>[^,]+),"
)

# Backward-compatible older client init line
CLIENT_ENABLE_OLD_RE = re.compile(
    r"Client (?P<cid>\d+): enabled, init_rank=(?P<init>\d+),"
)

CLIENT_RANK_UPDATE_RE = re.compile(
    r"Client (?P<cid>\d+): rank updated: "
    r"(?P<before>\d+) -> (?P<after>\d+)"
)

CLIENT_PRUNE_SUMMARY_RE = re.compile(
    r"Client (?P<cid>\d+): "
    r"candidates=(?P<cand>\d+)/(?P<before>\d+), "
    r"decreased=(?P<decreased>\d+), pruned=(?P<pruned>\d+), "
    r"new_rank=(?P<after>\d+)"
)

SERVER_RECEIVED_RE = re.compile(
    r"Received from client (?P<cid>\d+): .* rank=(?P<rank>\d+)"
)

SERVER_BROADCAST_RE = re.compile(
    r"Broadcasting to client (?P<cid>\d+): .* rank=(?P<rank>\d+)"
)


@dataclass
class Collector:
    current_round: Optional[int] = None
    rounds_seen: Set[int] = field(default_factory=set)
    client_ids: Set[int] = field(default_factory=set)

    default_init_rank: Optional[int] = None
    n_clients: Optional[int] = None
    rank_min: Optional[int] = None
    rank_max: Optional[int] = None

    # current round -> client -> rank
    events: Dict[int, Dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    client_init_ranks: Dict[int, int] = field(default_factory=dict)


def parse_adasparse_ranks_single_run(
    log_path: str,
    event_source: str = "both",
) -> Tuple[Dict[int, float], Dict[int, Dict[int, int]], Dict[str, object]]:
    event_source = event_source.lower().strip()
    if event_source not in {"client", "server", "both", "broadcast"}:
        raise ValueError("event_source must be one of: client, server, both, broadcast")

    c = Collector()

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_round = ROUND_START_RE.search(line)
            if m_round:
                rnd = int(m_round.group("round"))
                c.current_round = rnd
                c.rounds_seen.add(rnd)
                continue

            m_init = ADASPARSE_INIT_RE.search(line)
            if m_init:
                c.n_clients = int(m_init.group("n"))
                c.default_init_rank = int(m_init.group("init"))
                c.rank_min = int(m_init.group("min"))
                c.rank_max = int(m_init.group("max"))
                continue

            m_en = CLIENT_ENABLE_RE.search(line)
            if m_en:
                cid = int(m_en.group("cid"))
                init_r = int(m_en.group("init"))
                c.client_ids.add(cid)
                c.client_init_ranks[cid] = init_r
                continue

            m_en_old = CLIENT_ENABLE_OLD_RE.search(line)
            if m_en_old:
                cid = int(m_en_old.group("cid"))
                init_r = int(m_en_old.group("init"))
                c.client_ids.add(cid)
                c.client_init_ranks.setdefault(cid, init_r)
                continue

            if c.current_round is None:
                continue

            if event_source in {"client", "both"}:
                m_up = CLIENT_RANK_UPDATE_RE.search(line)
                if m_up:
                    cid = int(m_up.group("cid"))
                    after = int(m_up.group("after"))
                    c.client_ids.add(cid)
                    c.events[c.current_round][cid] = after
                    continue

                m_sum = CLIENT_PRUNE_SUMMARY_RE.search(line)
                if m_sum:
                    cid = int(m_sum.group("cid"))
                    after = int(m_sum.group("after"))
                    c.client_ids.add(cid)
                    c.events[c.current_round].setdefault(cid, after)
                    continue

            if event_source in {"server", "both"}:
                m_recv = SERVER_RECEIVED_RE.search(line)
                if m_recv:
                    cid = int(m_recv.group("cid"))
                    after = int(m_recv.group("rank"))
                    c.client_ids.add(cid)
                    c.events[c.current_round][cid] = after
                    continue

            if event_source == "broadcast":
                m_brd = SERVER_BROADCAST_RE.search(line)
                if m_brd:
                    cid = int(m_brd.group("cid"))
                    after = int(m_brd.group("rank"))
                    c.client_ids.add(cid)
                    c.events[c.current_round][cid] = after
                    continue

    if not c.rounds_seen and not c.events:
        return {}, {}, {"warning": "No rounds or rank events detected"}

    rounds = sorted(c.rounds_seen) if c.rounds_seen else sorted(c.events.keys())
    if not rounds:
        return {}, {}, {"warning": "No rounds detected"}

    clients = sorted(c.client_ids)
    if not clients and c.n_clients is not None:
        clients = list(range(1, c.n_clients + 1))

    # Seed round 0 using per-client initial ranks when available
    if 0 not in rounds:
        rounds = [0] + rounds

    for cid in clients:
        init_r = c.client_init_ranks.get(cid, c.default_init_rank)
        if init_r is not None:
            c.events[0].setdefault(cid, int(init_r))

    filled: Dict[int, Dict[int, int]] = {}
    last_rank: Dict[int, Optional[int]] = {
        cid: c.client_init_ranks.get(cid, c.default_init_rank) for cid in clients
    }

    for rnd in rounds:
        for cid, r_after in c.events.get(rnd, {}).items():
            last_rank[cid] = int(r_after)

        filled[rnd] = {}
        for cid in clients:
            if last_rank.get(cid) is not None:
                filled[rnd][cid] = int(last_rank[cid])

    avg: Dict[int, float] = {}
    for rnd in rounds:
        rs = list(filled[rnd].values())
        if rs:
            avg[rnd] = float(np.mean(rs))

    meta = {
        "n_clients": c.n_clients if c.n_clients is not None else len(clients),
        "default_init_rank": c.default_init_rank,
        "rank_min": c.rank_min,
        "rank_max": c.rank_max,
        "clients": clients,
        "rounds": rounds,
        "event_source": event_source,
        "client_init_ranks": dict(c.client_init_ranks),
    }

    return avg, filled, meta


def _parse_client_subset(s: Optional[str]) -> Optional[Set[int]]:
    if not s:
        return None
    out: Set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _infer_missing_client_init_ranks(
    clients: Set[int],
    client_ranks: Dict[int, Dict[int, int]],
    client_init_ranks: Dict[int, int],
) -> Dict[int, int]:
    init = dict(client_init_ranks)
    rounds = sorted(client_ranks.keys())
    first_round = rounds[0]
    for cid in clients:
        if cid not in init and cid in client_ranks.get(first_round, {}):
            init[cid] = int(client_ranks[first_round][cid])
    return init


def compute_savings_series(
    client_ranks: Dict[int, Dict[int, int]],
    *,
    client_init_ranks: Dict[int, int],
    clients: Set[int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Savings are measured against each client's own initial rank.

    active_pruned[r] =
        sum_c max(0, init_rank[c] - rank_c(r))

    cumulative_savings[r] =
        sum_{t <= r} active_pruned[t]
    """
    rounds_all = sorted(client_ranks.keys())
    active_pruned: Dict[int, int] = {}
    cumulative_savings: Dict[int, int] = {}

    cumulative = 0
    for rnd in rounds_all:
        total_pruned = 0
        for cid in clients:
            init_r = client_init_ranks[cid]
            rank = client_ranks.get(rnd, {}).get(cid, init_r)
            total_pruned += max(0, int(init_r) - int(rank))
        active_pruned[rnd] = total_pruned
        cumulative += total_pruned
        cumulative_savings[rnd] = cumulative

    return active_pruned, cumulative_savings


def summarize_savings(
    active_pruned: Dict[int, int],
    cumulative_savings: Dict[int, int],
    *,
    client_init_ranks: Dict[int, int],
    clients: Set[int],
) -> Dict[str, float]:
    rounds = sorted(active_pruned.keys())
    if not rounds:
        return {}

    n_rounds = len(rounds)
    baseline_per_round = float(sum(client_init_ranks[cid] for cid in clients))
    baseline_total = baseline_per_round * n_rounds
    final_active_pruned = float(active_pruned[rounds[-1]])
    final_cumulative = float(cumulative_savings[rounds[-1]])

    return {
        "baseline_per_round": baseline_per_round,
        "baseline_total": baseline_total,
        "final_active_pruned": final_active_pruned,
        "final_active_pruned_pct": (
            100.0 * final_active_pruned / baseline_per_round if baseline_per_round > 0 else 0.0
        ),
        "final_cumulative_savings": final_cumulative,
        "avg_saved_ranks_per_round": (
            final_cumulative / n_rounds if n_rounds > 0 else 0.0
        ),
        "overall_savings_pct": (
            100.0 * final_cumulative / baseline_total if baseline_total > 0 else 0.0
        ),
        "n_rounds": float(n_rounds),
    }


def plot_rank_evolution(
    avg_ranks: Dict[int, float],
    client_ranks: Dict[int, Dict[int, int]],
    *,
    client_init_ranks: Dict[int, int],
    title: str = "AdaSparse-LoRA Rank Evolution",
    output_path: Optional[str] = None,
    only_avg: bool = False,
    max_clients_in_legend: int = 20,
    client_subset: Optional[Set[int]] = None,
    round_from: Optional[int] = None,
    round_to: Optional[int] = None,
):
    if not avg_ranks:
        raise RuntimeError("No rank data to plot")

    rounds_all = sorted(avg_ranks.keys())
    rounds = list(rounds_all)
    if round_from is not None:
        rounds = [r for r in rounds if r >= round_from]
    if round_to is not None:
        rounds = [r for r in rounds if r <= round_to]
    if not rounds:
        raise RuntimeError("No rounds left after applying --from_round/--to_round filters")

    all_clients: Set[int] = set()
    for r in rounds_all:
        all_clients.update(client_ranks.get(r, {}).keys())
    clients = sorted(all_clients)
    if client_subset is not None:
        clients = [c for c in clients if c in client_subset]

    client_init_ranks = _infer_missing_client_init_ranks(set(clients), client_ranks, client_init_ranks)

    active_pruned_all, cumulative_savings_all = compute_savings_series(
        client_ranks,
        client_init_ranks=client_init_ranks,
        clients=set(clients),
    )

    savings_summary = summarize_savings(
        active_pruned_all,
        cumulative_savings_all,
        client_init_ranks=client_init_ranks,
        clients=set(clients),
    )

    # Restrict savings to shown rounds
    active_pruned = {r: active_pruned_all[r] for r in rounds if r in active_pruned_all}
    cumulative_savings = {r: cumulative_savings_all[r] for r in rounds if r in cumulative_savings_all}

    inferred_rank_min = min(client_init_ranks.values()) if client_init_ranks else 0

    fig, (ax_rank, ax_save) = plt.subplots(1, 2, figsize=(18, 7))
    fig.subplots_adjust(left=0.26, bottom=0.18, wspace=0.18)

    if not only_avg and clients:
        cmap = plt.colormaps.get_cmap("tab10")
        for idx, cid in enumerate(clients):
            xs, ys = [], []
            for r in rounds:
                v = client_ranks.get(r, {}).get(cid)
                if v is not None:
                    xs.append(r)
                    ys.append(v)
            if ys:
                ax_rank.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    color=cmap(idx % 10),
                    alpha=0.7,
                    label=f"Client {cid} r_init={client_init_ranks.get(cid, ys[0])}",
                )

    ax_rank.plot(
        rounds,
        [avg_ranks[r] for r in rounds],
        marker="s",
        markersize=6,
        linewidth=3,
        color="black",
        linestyle="--",
        label="Avg Rank",
        zorder=5,
    )

    ax_rank.set_xlabel("Training Round", fontsize=12)
    ax_rank.set_ylabel("Rank", fontsize=12)
    ax_rank.set_title(title, fontsize=14, fontweight="bold")
    ax_rank.grid(True, alpha=0.3)

    if len(rounds) > 25:
        step = max(1, len(rounds) // 10)
        ticks = rounds[::step]
        if rounds[-1] not in ticks:
            ticks = ticks + [rounds[-1]]
        ax_rank.set_xticks(ticks)
    else:
        ax_rank.set_xticks(rounds)

    save_color = "navy"
    ax_save.plot(
        rounds,
        [cumulative_savings[r] for r in rounds],
        marker="^",
        markersize=5,
        linewidth=2.5,
        color=save_color,
        alpha=0.95,
    )
    ax_save.set_title("Cumulative Savings", fontsize=12, fontweight="bold")
    ax_save.set_xlabel("Training Round", fontsize=11)
    ax_save.set_ylabel("Cumulative rank-round savings", fontsize=11)
    ax_save.grid(True, alpha=0.3)
    if len(rounds) > 25:
        ax_save.set_xticks(ticks)
    else:
        ax_save.set_xticks(rounds)

    # Keep the right axis well-behaved:
    # - minimum is always 0
    # - use integer ticks only
    # - keep the no-pruning case readable
    max_savings = max([cumulative_savings[r] for r in rounds], default=0)
    upper = max(1, int(math.ceil(max_savings * 1.05)))
    ax_save.set_ylim(0, upper)
    ax_save.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Stats block in the middle panel
    if savings_summary:
        stats_text = (
            f"Final active pruned ranks: {int(active_pruned[rounds[-1]])}. "
            f"Final active pruning ratio: {savings_summary['final_active_pruned_pct']:.2f}%. "
            f"Cumulative rank-round savings: {int(cumulative_savings[rounds[-1]])}. "
            f"Average saved ranks per round: {savings_summary['avg_saved_ranks_per_round']:.2f}. "
            f"Overall traffic reduction: {savings_summary['overall_savings_pct']:.2f}%."
        )

    handles, labels = ax_rank.get_legend_handles_labels()
    if not only_avg and len(labels) > (1 + max_clients_in_legend):
        kept_h, kept_l = [], []
        for i in range(min(max_clients_in_legend, len(handles) - 1)):
            kept_h.append(handles[i])
            kept_l.append(labels[i])
        kept_h.append(handles[-1])
        kept_l.append(labels[-1])
        handles, labels = kept_h, kept_l

    # Keep the main plot legend on the left side of the first subplot
    ax_rank.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.42, 0.995),
        fontsize=9,
        framealpha=0.9,
        borderaxespad=0.0,
    )

    # Put the savings statistics as bottom text under the right subplot
    if savings_summary:
        fig.text(
            0.5, 0.04, stats_text,
            va="bottom", ha="center",
            fontsize=10,
            wrap=True,
        )


    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot AdaSparse-LoRA rank evolution and cumulative savings "
            "(single-run logs)."
        )
    )

    parser.add_argument("--log_file", required=True, help="Path to exp_print.log")
    parser.add_argument("-o", "--output", default=None, help="Where to save the figure (optional)")
    parser.add_argument("-t", "--title", default="AdaSparse-LoRA Rank Evolution", help="Plot title")
    parser.add_argument(
        "--event_source",
        choices=["client", "server", "both", "broadcast"],
        default="both",
        help=(
            "Which rank signals to use (default: both). "
            "'client' and 'server' both use non-debug messages; "
            "'broadcast' uses debug broadcast lines."
        ),
    )
    parser.add_argument("--only_avg", action="store_true", help="Plot only the average rank curve")
    parser.add_argument(
        "--max_clients_in_legend",
        type=int,
        default=20,
        help="Max number of clients to show in legend (default: 20)",
    )
    parser.add_argument("--clients", default=None, help="Client subset, e.g. '1,2,5-8'")
    parser.add_argument("--from_round", type=int, default=None, help="Start round (inclusive)")
    parser.add_argument("--to_round", type=int, default=None, help="End round (inclusive)")

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: log file not found: {log_path}")
        sys.exit(1)

    output_dir = prepare_run_output_dir(log_path, 'rank_evolution_adasparse')
    console_log_path = output_dir / "console_output.txt"

    original_stdout = sys.stdout
    console_log_file = open(console_log_path, "w", encoding="utf-8")
    sys.stdout = TeeStdout(original_stdout, console_log_file)

    try:
        avg_ranks, client_ranks, meta = parse_adasparse_ranks_single_run(
            str(log_path),
            event_source=args.event_source,
        )

        if not avg_ranks:
            print(
                "Error: No AdaSparse-LoRA rank data found. "
                "Expected round markers and rank update / received-rank lines."
            )
            sys.exit(1)

        rounds = meta["rounds"]
        clients = meta["clients"]
        client_init_ranks = _infer_missing_client_init_ranks(
            set(clients), client_ranks, meta.get("client_init_ranks", {})
        )

        print(f"Parsed rounds: {min(rounds)}..{max(rounds)} (n={len(rounds)})")
        print(f"Parsed clients: {clients}")
        if meta.get("default_init_rank") is not None:
            print(f"Default init rank from server log: {meta['default_init_rank']}")
        if meta.get("rank_min") is not None and meta.get("rank_max") is not None:
            print(f"Rank bounds: [{meta['rank_min']}, {meta['rank_max']}]")

        client_init_desc = ", ".join(
            f"{cid}:{client_init_ranks[cid]}" for cid in sorted(client_init_ranks)
        )
        print(f"Per-client init ranks: {client_init_desc}")
        print(f"Output directory: {output_dir}")

        active_pruned_all, cumulative_savings_all = compute_savings_series(
            client_ranks,
            client_init_ranks=client_init_ranks,
            clients=set(clients),
        )
        summary = summarize_savings(
            active_pruned_all,
            cumulative_savings_all,
            client_init_ranks=client_init_ranks,
            clients=set(clients),
        )
        if summary:
            print(f"Final active pruned ranks: {int(summary['final_active_pruned'])}")
            print(
                "Final active pruning ratio: "
                f"{summary['final_active_pruned_pct']:.2f}% of initial-rank traffic"
            )
            print(
                "Cumulative rank-round savings: "
                f"{int(summary['final_cumulative_savings'])}"
            )
            print(
                "Average saved ranks per round: "
                f"{summary['avg_saved_ranks_per_round']:.2f}"
            )
            print(
                "Overall traffic reduction over the full run: "
                f"{summary['overall_savings_pct']:.2f}%"
            )

        output_path = args.output or str(output_dir / "adasparse_lora_rank_evolution.png")

        client_subset = _parse_client_subset(args.clients)

        plot_rank_evolution(
            avg_ranks,
            client_ranks,
            client_init_ranks=client_init_ranks,
            title=args.title,
            output_path=output_path,
            only_avg=args.only_avg,
            max_clients_in_legend=args.max_clients_in_legend,
            client_subset=client_subset,
            round_from=args.from_round,
            round_to=args.to_round,
        )
        print(f"Console log saved to: {console_log_path}")
    finally:
        sys.stdout = original_stdout
        console_log_file.close()

if __name__ == "__main__":
    main()
