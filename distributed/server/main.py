"""
Server orchestrator for distributed FedLoRA deployment.

Ties together the three permanent components on the server machine:
  1. ControlPlaneService — device registration + container launch (proto mode)
  2. ZMQServerCommManager — FL payload plane (relay or direct mode)
  3. FedLoRA Server — existing FL semantics with injected comm manager

Usage (relay mode — production, FL traffic through device_agent):
    python -m distributed.server.main \
        --config exp/distributed_fedit/config.yaml \
        --manifest distributed/configs/client_manifest.json \
        --host 0.0.0.0 \
        --proto \
        --device-port-offset 100

Usage (direct mode — workers connect directly to server FL ports):
    python -m distributed.server.main \
        --config ... --manifest ... --proto --direct
"""
import argparse
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from federatedscope.core.auxiliaries.logging import update_logger
import federatedscope.contrib.common as fs_common

import federatedscope.contrib.worker.adasparse_lora_worker
import federatedscope.contrib.worker.adasparse_lorav2_worker
import federatedscope.contrib.worker.adasparse_lorav3_worker
import federatedscope.contrib.worker.hetlora_worker
import federatedscope.contrib.worker.heterolora_worker
import federatedscope.contrib.worker.fah_qlora_worker

logger = logging.getLogger(__name__)


class ServerBandwidthShaper:
    """Server-side TC egress shaping, mirroring device_agent's limitBandwidth().

    Reads an AegisGov-format bandwidth_limits JSON and periodically applies
    TC rules on the server's egress interface via set_bandwidth.sh.
    Loops the trace when it reaches the end.
    """

    def __init__(self, json_path: str, interface: str,
                 script_path: str = None):
        import json as _json
        with open(json_path) as f:
            data = _json.load(f)
        self._limits = data["bandwidth_limits"]
        self._interface = interface
        if script_path is None:
            repo_root = Path(__file__).parent.parent.parent
            script_path = str(
                repo_root.parent
                / "AegisGov-master" / "scripts" / "set_bandwidth.sh"
            )
        self._script = script_path
        self._thread = None
        self._stop_event = None

    def start(self):
        import threading
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="server-tc-shaper"
        )
        self._thread.start()
        logger.info("Server TC shaper started: interface=%s, %d thresholds",
                     self._interface, len(self._limits))

    def stop(self):
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        import subprocess
        result = subprocess.run(
            ["sudo", "-n", "/usr/bin/bash", self._script,
             self._interface, "10000"],
            capture_output=True,
        )
        subprocess.run(
            ["sudo", "-n", "tc", "qdisc", "del", "dev",
             self._interface, "root"],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.warning(
                "TC cleanup may be incomplete; run: "
                "sudo tc qdisc del dev %s root", self._interface)
        logger.info("Server TC shaper stopped")

    def _run(self):
        import subprocess
        idx = 0
        start_time = time.time()
        while not self._stop_event.is_set():
            entry = self._limits[idx]
            target_time = start_time + entry["time"]
            now = time.time()
            if now >= target_time:
                mbps = entry["mbps"]
                subprocess.run(
                    ["sudo", "-n", "/usr/bin/bash", self._script,
                     self._interface, f"{mbps:.2f}"],
                    capture_output=True,
                )
                logger.debug("Server TC: set %s to %.2f Mbps", self._interface, mbps)

                if idx == len(self._limits) - 1:
                    logger.info("Server TC: trace ended, looping from start")
                    idx = 0
                    start_time = time.time()
                else:
                    idx += 1

                next_entry = self._limits[idx]
                sleep_secs = (start_time + next_entry["time"]) - time.time()
                if sleep_secs > 0:
                    self._stop_event.wait(sleep_secs)
            else:
                self._stop_event.wait(target_time - now)


def build_server(cfg, model, comm_manager, client_num, runner=None):
    from federatedscope.core.auxiliaries.worker_builder import get_server_cls

    server_cls = get_server_cls(cfg)
    logger.info("Dispatch: method=%s -> %s", cfg.federate.method, server_cls.__name__)

    device = "cpu"
    try:
        import torch
        if cfg.use_gpu and torch.cuda.is_available() and cfg.device >= 0:
            device = f"cuda:{cfg.device}"
    except Exception:
        pass

    kw = {}
    if runner is not None:
        if getattr(runner, 'fah_client_rank_caps', None):
            kw['fah_client_rank_caps'] = runner.fah_client_rank_caps
        if getattr(runner, 'hetero_lora_config', None) and getattr(runner, 'fah_enabled', False):
            kw['fah_cap_config_local'] = runner.hetero_lora_config

    server = server_cls(
        ID=0,
        state=0,
        config=cfg,
        data=None,
        model=model,
        client_num=client_num,
        total_round_num=cfg.federate.total_round_num,
        device=device,
        comm_manager=comm_manager,
        **kw,
    )
    return server


def main():
    parser = argparse.ArgumentParser(
        description="FedLoRA Distributed Server Orchestrator"
    )
    parser.add_argument("--config", required=True, help="Server YAML config")
    parser.add_argument("--manifest", required=True, help="Client manifest JSON")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port-offset", type=int, default=0)
    parser.add_argument("--proto", action="store_true",
                        help="Protobuf wire format for real device_agents")
    parser.add_argument("--direct", action="store_true",
                        help="Direct mode: workers connect to server FL ports "
                             "(bypass device_agent relay)")
    parser.add_argument("--device-port-offset", type=int, default=100,
                        help="Port offset used by device_agents "
                             "(--dev_system_port_offset).  Workers need this "
                             "to connect to the correct device_agent data "
                             "ports (60011/60012 + offset).  Ignored in "
                             "--direct mode.")
    parser.add_argument("--skip-control-plane", action="store_true",
                        help="Skip device registration; assume workers start externally")
    parser.add_argument("--device-wait-timeout", type=float, default=120.0,
                        help="Seconds to wait for device registration")
    parser.add_argument("--worker-config-path",
                        default=None,
                        help="Config path passed to workers (container-relative). "
                             "Defaults to --config if not specified.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--bandwidth-json", type=str, default="",
                        help="Path to server-side DL bandwidth_limits JSON "
                             "for TC egress shaping (enables server-side TC)")
    parser.add_argument("--bandwidth-setting", type=int, default=0,
                        help="Bandwidth profile index sent to device_agents "
                             "via SystemInfo (0=disabled, 1+=profile ID)")
    parser.add_argument("--server-nic", type=str, default="enp66s0f0",
                        help="Server network interface for TC shaping")
    parser.add_argument("--no-collect-logs", action="store_true",
                        help="Skip post-run worker log collection")
    parser.add_argument("opts", nargs="*", default=[],
                        help="Config overrides in KEY VALUE pairs, "
                             "e.g. federate.total_round_num 5")
    args = parser.parse_args()

    if args.worker_config_path is None:
        args.worker_config_path = args.config

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    relay_mode = not args.direct
    mode_str = "RELAY" if relay_mode else "DIRECT"
    logger.info("FL payload mode: %s", mode_str)

    # ── 1. Load config and model ─────────────────────────────────────────
    logger.info("Loading server config from %s", args.config)
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.auxiliaries.model_builder import get_model

    cfg = global_cfg.clone()
    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)

    try:
        cfg.defrost()
    except Exception:
        pass
    cfg.federate.mode = "standalone"
    if not cfg.eval.metrics:
        cfg.eval.metrics = ["acc"]

    # Port of standalone's _setup_base_quant() for the distributed path.
    # Without this the server's own get_model() falls through to FP32, so
    # the initial broadcast model diverges from what workers load.
    cfg.computation_quantization.method = 'none'
    # DeBERTa/GLUE runs use bf16 (nbits=16). LLM (Qwen2) runs MUST use fp32
    # (nbits=32): bf16 has no triu kernel on Volta (agxavier), so Qwen2's causal
    # mask crashes there. fp32 also keeps the server+worker initial models aligned.
    _is_llm = str(getattr(cfg.data, 'type', '')).endswith('@llm') or \
        str(getattr(getattr(cfg, 'trainer', None), 'type', '')).lower() == 'llmtrainer'
    cfg.computation_quantization.nbits = 32 if _is_llm else 16
    logger.info(
        "[BASE_QUANT] Distributed override: nbits=%d (%s)",
        cfg.computation_quantization.nbits,
        'fp32 for LLM/Qwen' if _is_llm else 'bf16 for GLUE')

    # The source yaml carries the worker's container-side model path
    # (/workspace/models/...). Translate it to the server-host path so the
    # server's get_model() can load weights from disk. Env var override takes
    # precedence for sites that keep models elsewhere.
    import os as _os
    _host_model_root = _os.environ.get(
        "FEDLORA_HOST_MODEL_ROOT", _os.path.expanduser("~/models")
    )
    _worker_prefix = "/workspace/models/"
    if cfg.model.type.startswith(_worker_prefix):
        _translated = cfg.model.type.replace(_worker_prefix, _host_model_root + "/", 1)
        logger.info(
            "[MODEL_PATH] Translating worker-side model path to host-side: %s -> %s",
            cfg.model.type, _translated,
        )
        cfg.model.type = _translated

    # Override manifest_path in adapter config to match --manifest flag.
    # YAMLs hardcode the 12-client manifest; sub-fleet runs need the group manifest.
    if hasattr(args, 'manifest') and args.manifest:
        if hasattr(cfg.glue, 'adapter') and hasattr(cfg.glue.adapter, 'manifest_path'):
            cfg.glue.adapter.manifest_path = args.manifest
        # LLM (Qwen) distributed runs key off llm.adapter, not glue.adapter; without
        # this the sub-fleet run silently falls back to the hardcoded 12-client manifest.
        if hasattr(cfg, 'llm') and hasattr(cfg.llm, 'adapter') and \
                hasattr(cfg.llm.adapter, 'manifest_path'):
            cfg.llm.adapter.manifest_path = args.manifest

    update_logger(cfg, clear_before_add=True)

    import os as _os
    _ext_fh = logging.FileHandler(
        _os.path.join(cfg.outdir, 'exp_print_extended.log'))
    _ext_fh.setLevel(logging.DEBUG)
    _ext_fh.setFormatter(logging.Formatter(
        "%(asctime)s (%(name)s:%(lineno)d) %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(_ext_fh)

    try:
        cfg.freeze()
    except Exception:
        pass

    client_num = cfg.federate.client_num
    total_rounds = cfg.federate.total_round_num
    logger.info("FL config: %d clients, %d rounds, method=%s",
                client_num, total_rounds, cfg.federate.method)

    logger.info("Building server model...")
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.worker_builder import get_server_cls
    data, _ = get_data(cfg.clone())

    # Run the same BaseRunner initialization that the standalone path uses.
    # This populates hetero_lora_config (per-client rank assignments from
    # distributed_fleet or other strategies) into cfg.*.adapter.hetero_ranks
    # .config_local, which HetLoRA/FAH/AdaSparse servers read at init time.
    # We subclass BaseRunner with no-op stubs for the abstract methods that
    # only matter for the standalone simulation loop.
    from federatedscope.core.fed_runner import BaseRunner

    class _DistributedRunnerInit(BaseRunner):
        def _set_up(self):
            pass
        def _get_server_args(self, resource_info, client_resource_info):
            return None, None, {}
        def _get_client_args(self, client_id, resource_info):
            return None, {}
        def run(self):
            raise RuntimeError("not used in distributed mode")

    server_cls = get_server_cls(cfg)
    runner = _DistributedRunnerInit(
        data=data,
        server_class=server_cls,
        client_class=server_cls,
        config=cfg,
    )
    cfg = runner.cfg
    config_local = fs_common.get_active_hetero_config_local(cfg)
    logger.info("BaseRunner initialized (hetero config populated, config_local=%s)",
                "present" if config_local else "absent")

    model = get_model(cfg, data)
    logger.info("Model built: %s", type(model).__name__)

    # ── 2. Load manifest ─────────────────────────────────────────────────
    import json
    with open(args.manifest) as f:
        manifest = json.load(f)
    manifest_clients = manifest.get("clients", [])
    logger.info("Manifest: %d clients", len(manifest_clients))

    # ── 3. Control plane (optional) ──────────────────────────────────────
    from distributed.control_plane.service import (
        ControlPlaneService,
        container_config_from_manifest_client,
    )

    ctrl = ControlPlaneService(
        host=args.host,
        port_offset=args.port_offset,
        manifest_path=args.manifest,
        proto_mode=args.proto,
    )
    if args.bandwidth_setting > 0:
        ctrl.bandwidth_setting = args.bandwidth_setting
        logger.info("Bandwidth shaping enabled: device_agents will use profile %d",
                     args.bandwidth_setting)

    if not args.skip_control_plane:
        ctrl.start()
        time.sleep(0.5)

        device_names = [c["device_name"] for c in manifest_clients]
        logger.info("Waiting for devices: %s (timeout=%.0fs)",
                     device_names, args.device_wait_timeout)
        if not ctrl.wait_for_devices(device_names, timeout=args.device_wait_timeout):
            logger.error("Timed out waiting for device registration. Exiting.")
            ctrl.stop()
            sys.exit(1)

        logger.info("All devices registered. Sending CONTAINER_START...")
        server_ip = args.host if args.host != "0.0.0.0" else _get_default_ip()
        for client_entry in manifest_clients:
            # In relay mode: workers connect to local device_agent ports.
            # In direct mode: workers connect directly to server FL ports.
            da_port_offset = (
                client_entry.get("da_port_offset",
                                 args.device_port_offset)
                if relay_mode else 0
            )
            cc = container_config_from_manifest_client(
                client_entry,
                config_path=args.worker_config_path,
                device_agent_host="127.0.0.1",
                device_agent_port_offset=da_port_offset,
            )
            cc.docker_tag = client_entry.get("image", "")

            if args.direct:
                # Direct mode: tell worker to connect to server FL ports
                cc.extra_args = {
                    "fl_server_host": server_ip,
                    "fl_server_port_offset": args.port_offset,
                    "verbose": args.verbose,
                }
                if args.opts:
                    cc.extra_args["config_opts"] = args.opts
            else:
                # Relay mode: no fl_server_host — worker uses device_agent
                cc.extra_args = {
                    "verbose": args.verbose,
                }

            if args.opts:
                cc.extra_args["config_opts"] = args.opts

            if config_local is not None:
                cid = int(client_entry["client_id"])
                client_key = fs_common.resolve_client_key(config_local, cid)
                if client_key and client_key in config_local:
                    cc.extra_args["client_rank_config"] = dict(config_local[client_key])
                    logger.info("Attached client_rank_config for %s (client %d)", client_key, cid)

            device_name = client_entry["device_name"]
            ctrl.send_container_start(device_name, cc)
            time.sleep(0.3)

        logger.info("CONTAINER_START sent. Waiting for workers to initialize...")
        time.sleep(5)

    # ── 4. FL payload plane ──────────────────────────────────────────────
    from distributed.comm import ZMQServerCommManager

    routing_table = ctrl.build_client_routing_table()
    logger.info("Client routing table: %s", routing_table)

    if relay_mode:
        # Relay mode: bridge through ControlPlaneService (no own sockets)
        server_comm = ZMQServerCommManager(
            host=args.host,
            port_offset=args.port_offset,
            client_routing_table=routing_table,
            direct_mode=False,
            ctrl_service=ctrl,
        )
    else:
        # Direct mode: bind own REP/PUB sockets
        server_comm = ZMQServerCommManager(
            host=args.host,
            port_offset=args.port_offset,
            client_routing_table=routing_table,
            direct_mode=True,
        )

    for cid, route in routing_table.items():
        server_comm.add_neighbors(
            neighbor_id=cid,
            address=route,
        )

    # ── 5. Build and run FedLoRA Server ──────────────────────────────────
    logger.info("Building FedLoRA Server with injected ZMQServerCommManager...")
    server = build_server(cfg, model, server_comm, client_num, runner=runner)
    logger.info("FedLoRA Server ready. Starting FL course...")

    # ── 5b. Server-side TC bandwidth shaping (optional) ─────────────────
    tc_shaper = None
    if args.bandwidth_json:
        tc_shaper = ServerBandwidthShaper(
            json_path=args.bandwidth_json,
            interface=args.server_nic,
        )
        tc_shaper.start()

    shutdown = False

    def _sigint_handler(sig, frame):
        nonlocal shutdown
        if shutdown:
            sys.exit(1)
        shutdown = True
        logger.info("Caught SIGINT — finishing current round then stopping")

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("FL course finished. Cleaning up...")
        if tc_shaper:
            tc_shaper.stop()
        try:
            server_comm.close()
        except Exception:
            pass
        ctrl.stop()

        if not args.no_collect_logs:
            _collect_worker_logs(cfg.outdir, args.manifest)

        logger.info("Server shutdown complete.")


def _collect_worker_logs(exp_dir: str, manifest_path: str):
    """Pull container logs from fleet devices into the experiment directory."""
    import subprocess
    script = Path(__file__).parent.parent.parent / \
        "1_scripts" / "distributed" / "log_tools" / "collect_logs.sh"
    if not script.exists():
        logger.warning("Log collection script not found: %s", script)
        return
    logger.info("Collecting worker logs into %s ...", exp_dir)
    try:
        result = subprocess.run(
            ["bash", str(script), exp_dir, manifest_path],
            capture_output=True, text=True, timeout=120,
        )
        for line in result.stdout.strip().splitlines():
            logger.info("  %s", line)
        if result.returncode != 0:
            logger.warning("Log collection exited %d: %s",
                           result.returncode, result.stderr.strip())
    except subprocess.TimeoutExpired:
        logger.warning("Log collection timed out after 120s")
    except Exception as e:
        logger.warning("Log collection failed: %s", e)


def _get_default_ip():
    """Best-effort: get a routable IP for this machine."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    main()
