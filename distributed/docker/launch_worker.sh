#!/bin/bash
# FedLoRA worker launcher for AegisGov device_agent integration.
#
# AegisGov's runCompose() cannot pass EXECUTABLE with spaces (unquoted
# in the system() call).  This script is set as EXECUTABLE (single token,
# no spaces) and reads the actual worker args from START_STRING (JSON,
# properly single-quoted by runCompose).

set -euo pipefail

exec python3 -c "
import json, os, sys

cfg = json.loads(os.environ.get('START_STRING', '{}'))
args = [sys.executable, '-m', 'distributed.worker.main']

cid = cfg.get('client_id', -1)
if cid >= 0:
    args.extend(['--client-id', str(cid)])

cn = cfg.get('container_name', '')
if cn:
    args.extend(['--container-name', cn])

pp = cfg.get('partition_path', '')
if pp:
    args.extend(['--partition-path', pp])

cp = cfg.get('config_path', '')
if cp:
    args.extend(['--config', cp])

dah = cfg.get('device_agent_host', '127.0.0.1')
args.extend(['--device-agent-host', dah])

dapo = cfg.get('device_agent_port_offset', 0)
args.extend(['--device-agent-port-offset', str(dapo)])

extra = cfg.get('extra_args', {})
if isinstance(extra, dict):
    fh = extra.get('fl_server_host', '')
    if fh:
        args.extend(['--fl-server-host', fh])
    fpo = extra.get('fl_server_port_offset', 0)
    if fpo:
        args.extend(['--fl-server-port-offset', str(fpo)])
    crc = extra.get('client_rank_config', '')
    if crc:
        import json as _json
        args.extend(['--client-rank-config', _json.dumps(crc) if isinstance(crc, dict) else str(crc)])
    if extra.get('verbose'):
        args.append('--verbose')
    config_opts = extra.get('config_opts', [])
    if config_opts:
        args.extend(['--config-opts', json.dumps(config_opts)])

nic = cfg.get('nic', '')
if nic:
    args.extend(['--nic', nic])
if cfg.get('shared_nic', False):
    args.append('--shared-nic')

print(f'[launch_worker] exec: {\" \".join(args)}', flush=True)
os.execvp(args[0], args)
"
