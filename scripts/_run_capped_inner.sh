#!/usr/bin/env bash
# Inner half of run_capped.sh. Lives in its own file (not an inline `bash -c '...'` string)
# because systemd-run performs its OWN $VAR / ${VAR} specifier expansion on the literal command
# line text it's given, before any shell ever sees it -- an inline string with $cg_path in it
# gets silently rewritten to an empty string by systemd itself first. Reading the variables from
# a real file sidesteps that entirely: systemd never looks inside files the command happens to
# execute.
set -euo pipefail
cg_path=$(awk -F: '{print $3}' /proc/self/cgroup)
export QWEN_GLOBAL_MEMORY_CGROUP="/sys/fs/cgroup${cg_path}"
max=$(cat "$QWEN_GLOBAL_MEMORY_CGROUP/memory.max")
swapmax=$(cat "$QWEN_GLOBAL_MEMORY_CGROUP/memory.swap.max")
# Read the watchdog limit back from the cgroup's actual memory.max rather than recomputing it
# from the --ram-gb number: systemd-run's "G" suffix is binary (GiB = 1024^3) while a naive
# `ram_gb * 1_000_000_000` is decimal GB, so the two disagree by ~7% and the HostMemory assert
# (cap <= limit) fails the moment they don't match exactly. Single source of truth: memory.max.
export QWEN_RAM_LIMIT_BYTES="$max"
echo "run_capped.sh: cgroup=$QWEN_GLOBAL_MEMORY_CGROUP memory.max=$max memory.swap.max=$swapmax QWEN_RAM_LIMIT_BYTES=$QWEN_RAM_LIMIT_BYTES" >&2
exec "$@"
