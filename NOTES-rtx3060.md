# Adapting lean-qwenflash for a 12 GB RTX 3060 (WSL2, desktop-shared GPU)

Status as of 2026-10-26: **code fixes verified in isolation; full end-to-end load NOT attempted**
(the 85 GB model isn't downloaded and the store isn't packed yet -- see "Blocked" below).
Nothing here is a claim that this works on 12 GB. It's the groundwork plus an honest list of
what's still unknown.

## What was fixed (verified against the real hardware)

Upstream (`MaxDevv/lean-qwenflash`) was built and benchmarked on an RTX 4090 intentionally capped
to 16 GB. Two of its hardcoded assumptions break outright on a smaller/different card:

1. **`qwen_offload.py`: `torch.cuda.set_per_process_memory_fraction(15_000_000_000 / total)`**
   On a 12,288 MiB (12.88 GB decimal) card this fraction is ~1.16, and PyTorch raises `ValueError`
   for any fraction > 1 -- this crashed before anything else ran. Replaced with a budget derived
   from `torch.cuda.mem_get_info()` at call time (so it accounts for whatever the desktop
   compositor / other GPU clients already hold), minus headroom, clamped to `(0.05, 0.97]`.
   Tested standalone against the real GPU: `total=12.88GB free=11.79GB headroom=1.55GB
   fraction=0.795` -- no exception.

2. **`qwen_offload.py`: `TORCH_CUDA_ARCH_LIST` defaulted to `'8.9'`** (Ada / RTX 4090). On any
   other architecture the compiled kernels are missing for the device and fail at first kernel
   launch. Now derived from `torch.cuda.get_device_capability()`. Tested standalone: RTX 3060 ->
   `capability=(8, 6)` -> `TORCH_CUDA_ARCH_LIST=8.6`, which matches the value the project's own
   vendored GLM-lineage benchmark script (`qwen38-flash-offload/results/.../p003_ab.sh`) already
   uses for Ampere.

3. **`memory_guard.py`: `DeviceMemory` hardcoded its watchdog limit to 16 GB**, so on a card
   smaller than 16 GB the watchdog would never fire before a real CUDA OOM (NVML reports
   whole-device usage under WDDM, so this was never going to be "our process's" 16 GB anyway).
   Now reads `QWEN_VRAM_LIMIT_BYTES`, consistent with what `qwen_offload.py` actually configures.
   **Also fixed:** the torch fraction budget (#1) and this watchdog were computed independently
   and could land within ~0.17 GB of each other -- any non-torch VRAM growth (CUDA context,
   another app) would then trip `os._exit(86)` even though torch stayed under its own cap. The
   torch budget is now also bounded by `QWEN_VRAM_LIMIT_BYTES - already_used - 0.5GB` margin.
   Verified: with `QWEN_VRAM_LIMIT_BYTES=11.5GB`, the real RTX 3060 now computes
   `fraction=0.769` (projected device usage 11.0 GB, a full 0.5 GB under the 11.5 GB watchdog),
   vs. the unbounded version's 0.17 GB margin before this fix.

4. **`memory_guard.py: HostMemory` requires a cgroup with a real `memory.max` and
   `memory.swap.max == 0`.** Neither holds in a plain WSL2 shell on this machine (root cgroup is
   uncapped; `.wslconfig` sets `swap=16GB`). Added `scripts/run_capped.sh` + `_run_capped_inner.sh`:
   a `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0` wrapper that resolves its
   own cgroup path at runtime and exports `QWEN_GLOBAL_MEMORY_CGROUP` / `QWEN_RAM_LIMIT_BYTES`
   correctly (reading the limit back from the cgroup's actual `memory.max`, not recomputed
   separately -- an earlier draft had a binary-GiB vs decimal-GB unit mismatch that failed the
   same assert it was trying to satisfy). **Verified end-to-end** with a real `HostMemory(...)`
   construction inside the wrapper: `limit=2147483648 current=6737920`, closed cleanly.

Added `profiles/rtx3060-12gb.env`: a conservative first-run config (elastic/staging cache off,
small `--cache 16384` context, `QWEN38_EC_MAX_SLOTS=256`, `QWEN_VRAM_LIMIT_BYTES=11.5GB`) to try
before touching any of the richer cache-tuning knobs, per the vendored engine's own
`qwen38-offload-setup/SKILL.md` ("tune one setting at a time against a control"). The
`run_capped.sh` example in that file now passes a *larger* `MemoryMax` than `--ram-gb`
(`run_capped.sh 20 ... --ram-gb 8`, not `9 ... --ram-gb 9`): `--ram-gb` sizes only the expert RAM
cache, while the cgroup also has to hold Python, torch, the CUDA host runtime, and the pinned
staging rings -- the README's own 3080-laptop row needed 19-24 GB of system RAM for an 8 GB cache.

**Pinned host memory under WSL2: tested, not a blocker.** The RAM tier and NVMe->GPU staging both
depend on multi-GB pinned (`cudaHostAlloc`) buffers, and CUDA-on-WSL's documented limitations
include restricted pinned memory in some configurations, so this was checked directly rather than
assumed: inside `run_capped.sh 20 --`, `torch.empty(n, dtype=torch.uint8, pin_memory=True)`
succeeded at 2 GB, 4 GB, and 8 GB (1.78s / 3.49s / 7.03s respectively) with no error. WSL2 itself
is not the blocker here.

**The README's build step doesn't exist.** `python compile_local.py` (README step 2) refers to a
file that isn't in the repo -- confirmed by `find` turning up nothing, and by upstream's own open
issue, filed the same day: `MaxDevv/lean-qwenflash#1 "No compile_local.py in repo"`. The real
build path is `qwen38-flash-offload/scripts/install.sh`: it needs `nvcc` + `ninja` (no GPU needed
to build) and JIT-compiles each native extension via `torch.utils.cpp_extension.load()` through
every module's own lazy `_EXT` loader (`expert_cache._ext()`, `nv_tier._ext()`, `nv2.ext()`,
`cpu_tier._build()`, `k_hcfuse.ext()`, the fast-split kernel). **Do not set `QWEN_PREBUILT_EXT=1`**
-- `qwen_offload.py` has an opt-in path that loads prebuilt `.so` files from
`qwen38-flash-offload/build/`, but those were built for the author's torch ABI and `sm_89`
(4090); loading them here would be silent ABI/arch mismatch, not a working shortcut.

## What's still unverified

- **Whether the dense (non-expert) model fits at all.** The HF revision `3.05bpw_h5_ng5` totals
  **85.14 GB** across 25 files (confirmed via the HF tree API, read-only, nothing downloaded) --
  this matches the top-level README's "~85 GB", not the vendored SKILL.md's "125.3 GB" (that
  number is from the GLM-lineage doc and doesn't apply here). Of that, `ngram_embedding.safetensors`
  (32.64 GB) streams from disk per the README and the 7 model shards total ~52.35 GB. The README
  says the packed *expert* store alone is ~47 GB, which would put the *dense* (non-expert) part
  at roughly **52.35 - 47 ≈ 5.3 GB** -- an estimate from subtraction, not a measurement. If the
  real dense footprint plus KV cache plus framework overhead exceeds the ~9-10 GB actually free
  on this card (12.88 GB total, minus ~1-3 GB already held by the desktop compositor and whatever
  else is running, minus CUDA context), **no amount of tuning fixes that** -- it's a hard capacity
  question that can only be answered by actually loading the model.
- **Whether `QWEN38_EC_MAX_SLOTS=256` and the other `profiles/rtx3060-12gb.env` values are even
  in the right range.** They're derived from the engine's own formulas and the upstream SKILL.md's
  "keep EC_MAX_SLOTS=1376 on a desktop-shared GPU" note (scaled down for 12 GB vs. that note's
  24 GB target), not from a measurement on this card.
- Host RAM available for `--ram-gb` hasn't been re-checked against whatever else is running right
  now (`.wslconfig` caps the whole WSL VM at `memory=40GB`; that's a ceiling, not a guarantee of
  headroom).

## Blocked: where does ~90-130 GB of data actually live?

Checked before downloading anything:

- **C: drive has only 45 GB free** (96% full). The WSL2 root filesystem (`ext4.vhdx`) already
  consumes **177 GB** on C:, and WSL reports its own virtual filesystem as "813 GB available" --
  that number is **not real headroom**; the vhdx is a dynamically-growing file on the actual,
  nearly-full C: drive. Writing the model/store into the default WSL root risks filling C:
  entirely. Do not do this.
- **D: drive (SK hynix PC801 NVMe) has 237 GB free**, but it's mounted via `drvfs` (`/mnt/d`),
  which the project's own setup guidance explicitly excludes for the store ("Not suitable: ...
  network filesystems, FUSE" -- `drvfs` has the same O_DIRECT-reliability problem). **Correction
  to an earlier draft of this note:** D: is NOT fine for the model directory either, only for the
  initial download before it's moved. `qwen_offload.py` passes `embed_disk=True` and sets
  `EXL3_NGRAM_STREAM=1`; the `assert ... mode == 'trellis_disk'` means the 32.64 GB
  `ngram_embedding.safetensors` streams directly from wherever `--model` points **during every
  request**, not just once at load. So both `--model` and `--store` need to sit on the same
  native O_DIRECT-capable ext4, not just the packed store -- roughly 85 GB (model, including the
  n-gram file) + 47 GB (store) if both are kept side by side, or ~52 GB if the model dir is
  trimmed to just the shards actually needed post-packing. Either way, that's more than D:'s 237
  GB would need to worry about as a ceiling, but it rules out leaving the model on `/mnt/d` long
  term.
- **Only two physical NVMes currently show up in Windows** (`Get-PhysicalDisk`): the Intel
  `SSDPEKNW010T8` (C:, and notably a QLC Gen3 drive, ~1.8 GB/s class -- below the project's
  3+ GB/s floor even if it had space) and the SK hynix `PC801` (D:). **The "spare NVMe drive" is
  not currently attached to this machine.** Getting real O_DIRECT performance means installing it
  and attaching it to WSL2 as a raw block device (`wsl --mount --bare`), then formatting it
  **inside WSL** as ext4 -- not through a Windows drive letter. That's the setup the original
  request ("I have a spare NVMe drive") was already pointing at; it just isn't installed yet.

## Environment gaps (not yet addressed, need a decision first)

- Python here is 3.14.4; the README wants 3.10-3.12. Needs a venv (e.g. via `uv`).
- `nvcc` / CUDA toolkit isn't installed in this WSL distro (the real build step,
  `qwen38-flash-offload/scripts/install.sh`, needs it). This needs `apt` + sudo. **Correction to
  an earlier draft of this note:** the toolkit major version must match whatever torch build goes
  into the venv, not be guessed independently. The torch already present on this machine is
  `2.13.0+cu130` (driver 616.56 supports it), and `torch.utils.cpp_extension` refuses to build
  when `nvcc`'s CUDA major version differs from torch's -- so this points at
  `cuda-toolkit-13-0`, not `12-x`, *if* the venv ends up using the same cu130 wheel. Confirm the
  target torch version before installing the toolkit. Never install the `cuda` meta-package or
  any in-WSL `nvidia-driver-*` package -- those break GPU passthrough.
- Anything that defaults to `~/.cache` (notably `huggingface_hub`'s default cache) lands on C:,
  which has ~45 GB of real free space. Point `HF_HOME` (and `download_http.py --model-dir`) at
  whichever volume ends up hosting the model, not the default.

## Three possible disk routes (none set up yet)

1. **Install the spare NVMe, attach it to WSL2 raw** (`wsl --mount --bare`), format ext4 inside
   WSL. The original ask. Needs a free M.2 slot and a physical install.
2. **A fixed-size VHDX on D:**, created with `diskpart`, attached with `wsl --mount --vhd` (check
   exact flags via `wsl --help` on this WSL version, 2.7.3.0), then `mkfs.ext4` inside WSL. No
   hardware install needed, and D: (PC801, Gen4) is a faster drive than the Intel 660p on C:
   anyway. O_DIRECT throughput through this path is unmeasured -- probe it with `dd oflag=direct`
   (write) and `iflag=direct` (read) before trusting it for the real store.
3. Do nothing extra and accept `/mnt/d`'s drvfs limitations -- ruled out above for the reasons
   already covered (store + streamed n-gram table both need real O_DIRECT).

Routes 1 and 2 both need an admin PowerShell window and **must be redone after every
`wsl --shutdown` or reboot** (the mount doesn't persist on its own).

## Next steps (held pending Stuart's input)

1. Is the spare NVMe physically installed in this machine yet, or is there even a free M.2 slot
   for it -- or would the VHDX-on-D: route (above) be preferable regardless?
2. Which torch/CUDA combination should the fresh 3.10-3.12 venv target, so the right
   `cuda-toolkit-1X-Y` apt package (sudo required) gets installed once, not guessed twice?
3. Once 1+2 are answered: set up the chosen disk route, run
   `qwen38-flash-offload/scripts/install.sh` to build the native extensions, resolve the real
   dense-weight footprint by actually downloading and loading the model (the only way to answer
   the capacity question above), then work through
   `qwen38-flash-offload/skills/qwen38-offload-setup/SKILL.md` step by step, starting from
   `profiles/rtx3060-12gb.env`.
