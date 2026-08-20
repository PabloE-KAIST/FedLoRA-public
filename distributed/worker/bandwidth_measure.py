"""OS-level bandwidth measurement for distributed FL workers.

Measures actual upload/download throughput by reading NIC byte counters
from /sys/class/net/<iface>/statistics/{tx_bytes,rx_bytes} and computing
delta bytes / delta time.

For devices that share a NIC (x86-worker virtual devices), NIC counters
reflect combined traffic from all co-located workers. In that case,
the module falls back to payload-size + wall-clock timing of the ZMQ
send/recv operations.

Usage in ZMQClientCommManager:
    bm = BandwidthMeasure(nic="eno1")
    bm.mark_upload_start()
    # ... send payload ...
    bm.mark_upload_end(payload_bytes=len(serialized))
    bm.mark_download_start()
    # ... receive payload ...
    bm.mark_download_end(payload_bytes=len(received))
    metrics = bm.get_round_metrics(round_idx)
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SYS_NET_DIR = "/sys/class/net"


@dataclass
class RoundCommMetrics:
    round_idx: int
    upload_bytes: int = 0
    download_bytes: int = 0
    upload_wall_seconds: float = 0.0
    download_wall_seconds: float = 0.0
    upload_nic_delta_bytes: int = 0
    download_nic_delta_bytes: int = 0
    measured_ul_kbps: float = 0.0
    measured_dl_kbps: float = 0.0
    nic_available: bool = False


class BandwidthMeasure:
    """Lightweight per-round bandwidth measurement for a single NIC."""

    def __init__(self, nic: str = "", shared_nic: bool = False):
        self._nic = nic
        self._shared_nic = shared_nic
        self._nic_available = False

        if nic and not shared_nic:
            tx_path = os.path.join(_SYS_NET_DIR, nic, "statistics", "tx_bytes")
            if os.path.exists(tx_path):
                self._nic_available = True
                logger.info("BandwidthMeasure: NIC %s available for stats", nic)
            else:
                logger.info(
                    "BandwidthMeasure: NIC %s stats not available, "
                    "falling back to timing-based measurement", nic
                )

        # Current phase timestamps
        self._ul_start_time: float = 0.0
        self._ul_start_tx: int = 0
        self._dl_start_time: float = 0.0
        self._dl_start_rx: int = 0

        # Completed round metrics
        self._current_round: int = -1
        self._current_metrics: Optional[RoundCommMetrics] = None
        self._history: List[RoundCommMetrics] = []

    def _read_tx_bytes(self) -> int:
        if not self._nic_available:
            return 0
        try:
            path = os.path.join(_SYS_NET_DIR, self._nic, "statistics", "tx_bytes")
            with open(path) as f:
                return int(f.read().strip())
        except (IOError, ValueError):
            return 0

    def _read_rx_bytes(self) -> int:
        if not self._nic_available:
            return 0
        try:
            path = os.path.join(_SYS_NET_DIR, self._nic, "statistics", "rx_bytes")
            with open(path) as f:
                return int(f.read().strip())
        except (IOError, ValueError):
            return 0

    def begin_round(self, round_idx: int) -> None:
        if self._current_metrics is not None:
            self._history.append(self._current_metrics)
        self._current_round = round_idx
        self._current_metrics = RoundCommMetrics(
            round_idx=round_idx,
            nic_available=self._nic_available,
        )

    def mark_upload_start(self) -> None:
        self._ul_start_time = time.monotonic()
        self._ul_start_tx = self._read_tx_bytes()

    def mark_upload_end(self, payload_bytes: int = 0) -> None:
        elapsed = time.monotonic() - self._ul_start_time
        tx_delta = self._read_tx_bytes() - self._ul_start_tx

        if self._current_metrics is None:
            return

        self._current_metrics.upload_bytes = payload_bytes
        self._current_metrics.upload_wall_seconds = elapsed
        self._current_metrics.upload_nic_delta_bytes = tx_delta

        if elapsed > 0:
            if self._nic_available and tx_delta > 0:
                self._current_metrics.measured_ul_kbps = (
                    tx_delta * 8 / elapsed / 1000
                )
            elif payload_bytes > 0:
                self._current_metrics.measured_ul_kbps = (
                    payload_bytes * 8 / elapsed / 1000
                )

        logger.debug(
            "Upload: %d bytes, %.3fs, NIC delta=%d bytes, "
            "measured=%.1f kbps",
            payload_bytes, elapsed, tx_delta,
            self._current_metrics.measured_ul_kbps,
        )

    def mark_download_start(self) -> None:
        self._dl_start_time = time.monotonic()
        self._dl_start_rx = self._read_rx_bytes()

    def mark_download_end(self, payload_bytes: int = 0) -> None:
        elapsed = time.monotonic() - self._dl_start_time
        rx_delta = self._read_rx_bytes() - self._dl_start_rx

        if self._current_metrics is None:
            return

        self._current_metrics.download_bytes = payload_bytes
        self._current_metrics.download_wall_seconds = elapsed
        self._current_metrics.download_nic_delta_bytes = rx_delta

        if elapsed > 0:
            if self._nic_available and rx_delta > 0:
                self._current_metrics.measured_dl_kbps = (
                    rx_delta * 8 / elapsed / 1000
                )
            elif payload_bytes > 0:
                self._current_metrics.measured_dl_kbps = (
                    payload_bytes * 8 / elapsed / 1000
                )

        logger.debug(
            "Download: %d bytes, %.3fs, NIC delta=%d bytes, "
            "measured=%.1f kbps",
            payload_bytes, elapsed, rx_delta,
            self._current_metrics.measured_dl_kbps,
        )

    def get_round_metrics(self, round_idx: Optional[int] = None) -> Optional[RoundCommMetrics]:
        if round_idx is None:
            return self._current_metrics
        for m in self._history:
            if m.round_idx == round_idx:
                return m
        if self._current_metrics and self._current_metrics.round_idx == round_idx:
            return self._current_metrics
        return None

    def get_all_metrics(self) -> List[RoundCommMetrics]:
        result = list(self._history)
        if self._current_metrics is not None:
            result.append(self._current_metrics)
        return result

    def to_dict(self, round_idx: Optional[int] = None) -> Dict:
        m = self.get_round_metrics(round_idx)
        if m is None:
            return {}
        return {
            "round": m.round_idx,
            "upload_bytes": m.upload_bytes,
            "download_bytes": m.download_bytes,
            "upload_wall_seconds": round(m.upload_wall_seconds, 4),
            "download_wall_seconds": round(m.download_wall_seconds, 4),
            "upload_nic_delta_bytes": m.upload_nic_delta_bytes,
            "download_nic_delta_bytes": m.download_nic_delta_bytes,
            "measured_ul_kbps": round(m.measured_ul_kbps, 2),
            "measured_dl_kbps": round(m.measured_dl_kbps, 2),
            "nic": self._nic,
            "nic_available": m.nic_available,
        }
