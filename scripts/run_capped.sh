#!/usr/bin/env bash
# Launch wrapper that satisfies memory_guard.HostMemory's hard assertions on this machine.
#
# Why this exists: HostMemory (memory_guard.py) asserts the cgroup it's pointed at has a real
# memory.max (not "max") and memory.swap.max == 0. On this box neither holds by default:
#   - the WSL2 root cgroup's memory.max is "max" (uncapped) unless something already scoped it;
#   - .wslconfig sets swap=16GB, so a plain shell's cgroup has swap available.
# Both qwen_server.py and infer.py require QWEN_GLOBAL_MEMORY_CGROUP to point at a cgroup that
# passes those asserts -- infer.py's fallback ('/sys/fs/cgroup') is the uncapped root and will
# fail the first assert. This wrapper creates a scope that satisfies both, computes its own
# cgroup path from the inside (so it works regardless of the scope name systemd picks), and
# execs the real command inside it.
#
# Verified on this machine 2026-10-26: systemd is PID 1, cgroup v2 is mounted, and
# `systemd-run --user --scope -p MemoryMax=200M -p MemorySwapMax=0 ...` produced a scope with
# memory.max=209715200 and memory.swap.max=0 exactly as expected. The actual qwen_offload.py
# load path has NOT been run through this wrapper yet (no model on disk).
#
# Usage:
#   scripts/run_capped.sh <ram-gb> -- <command> [args...]
# Example:
#   set -a; source profiles/rtx3060-12gb.env; set +a
#   scripts/run_capped.sh 20 -- python3 qwen_server.py --model ./model --store ./store/qwen.bin \
#       --ram-gb 8 --cache "$QWEN_CACHE_TOKENS" --chunk 8192 --port 30000
# (<ram-gb> here is the whole cgroup's MemoryMax, not --ram-gb's expert-cache size -- give it
# headroom for python/torch/CUDA/staging on top of the cache; see profiles/rtx3060-12gb.env.)
set -euo pipefail

RAM_GB="${1:?usage: run_capped.sh <ram-gb> -- <command> [args...]}"
shift
if [[ "${1:-}" == "--" ]]; then
  shift
else
  echo "run_capped.sh: expected '--' before the command, got '${1:-}'" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "run_capped.sh: no command given after '--'" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec systemd-run --user --scope --quiet \
  -p "MemoryMax=${RAM_GB}G" -p MemorySwapMax=0 \
  bash "${SCRIPT_DIR}/_run_capped_inner.sh" "$@"
