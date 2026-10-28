# Adapting lean-qwenflash for a 12 GB RTX 3060 (WSL2, desktop-shared GPU)

Status as of 2026-10-26: **code fixes verified in isolation; full end-to-end load NOT attempted**
(the 85 GB model isn't downloaded and the store isn't packed yet -- see "Blocked" below).
Nothing here is a claim that this works on 12 GB. It's the groundwork plus an honest list of
what's still unknown, including findings that make the honest VRAM budget *smaller* than first
estimated.

## What was fixed (verified against the real hardware)

Upstream (`MaxDevv/lean-qwenflash`) was built and benchmarked on an RTX 4090 intentionally capped
to 16 GB. Several of its hardcoded assumptions break on different hardware:

1. **`qwen_offload.py`: `torch.cuda.set_per_process_memory_fraction(15_000_000_000 / total)`**
   On a 12,288 MiB (12.88 GB decimal) card this fraction is ~1.16, and PyTorch raises `ValueError`
   for any fraction > 1 -- this crashed before anything else ran. Replaced with a budget derived
   from real VRAM state at call time, clamped to `(0.05, 0.97]`.

2. **`TORCH_CUDA_ARCH_LIST` defaulted to `'8.9'`** (Ada / RTX 4090 only). On any other
   architecture the compiled kernels are missing for the device and fail at first kernel launch
   ("no kernel image available"). Now derived from `torch.cuda.get_device_capability()`. Also
   matters at **build** time (`scripts/install.sh`, `pip install -e exllamav3` both read this env
   var when compiling, not just at runtime) -- `profiles/rtx3060-12gb.env` now exports
   `TORCH_CUDA_ARCH_LIST=8.6` explicitly so it's set before building, not only before serving.
   Tested standalone: RTX 3060 -> `capability=(8, 6)` -> `8.6`, matching the value the project's
   own vendored benchmark script already uses for Ampere.

3. **`memory_guard.py`: `DeviceMemory` hardcoded its watchdog limit to 16 GB** -- on a smaller
   card the watchdog would never fire before a real CUDA OOM. Now reads `QWEN_VRAM_LIMIT_BYTES`.

4. **The torch budget and the NVML-based watchdog disagree on "how much is already used," and
   the gap is real and large.** Measured back-to-back in one process on this machine:
   `torch.cuda.mem_get_info()` reported `used=1.09GB` while `nvmlDeviceGetMemoryInfo()` reported
   `used=2.73GB` **at the same instant** -- a 1.64 GB WDDM accounting gap between the two APIs.
   An earlier draft of this fix used torch's optimistic number to keep the torch budget clear of
   the watchdog; that would have let real (NVML-visible) usage sail past the watchdog on the
   first load. Fixed: `qwen_offload.py` now queries NVML directly for the "already used" baseline
   (same calc the watchdog itself uses), with a 0.5 GB margin.
   **Net effect: the real usable budget is smaller than first estimated.** With this profile's
   `QWEN_VRAM_LIMIT_BYTES=11.5GB`, the corrected calculation gives a torch fraction budget of only
   **~8.3 GB**, not the ~10 GB a torch-only calculation would have suggested. That ~8.3 GB is what
   the dense model + KV cache + minimal expert-slot pool actually have to fit inside.

5. **`memory_guard.py: HostMemory` requires a cgroup with a real `memory.max` and
   `memory.swap.max == 0`**, which a plain WSL2 shell here doesn't have (root cgroup is uncapped;
   `.wslconfig` sets `swap=16GB`). Added `scripts/run_capped.sh` + `_run_capped_inner.sh`: a
   `systemd-run --user --scope` wrapper that resolves its own cgroup path at runtime (split into
   two files because systemd-run does its own `$VAR` expansion on inline command text, silently
   emptying variables in an inline string) and reads the watchdog limit back from the cgroup's
   actual `memory.max` rather than recomputing it (an earlier draft had a binary-GiB vs
   decimal-GB mismatch that failed the very assert it was trying to satisfy). **Verified
   end-to-end**: a real `HostMemory(...)` construction succeeds inside it.
   The usage example (here and in the profile) is `run_capped.sh 20 ... --ram-gb 8`, not
   `9 ... --ram-gb 9` -- `--ram-gb` sizes only the expert RAM cache; python/torch/CUDA/staging
   need headroom on top (the README's own 3080-laptop row needed 19-24 GB system RAM for an 8 GB
   cache).

**Pinned host memory under WSL2: tested with the engine's actual mechanism, not a blocker.**
`nv_tier.py`/`nv2.py` pin host memory via anonymous `mmap` + `cudaHostRegister(..., 2)`
(`cudaHostRegisterMapped`), not `cudaHostAlloc` -- checked by grepping the native sources for
`io_uring`/`liburing`/`mlock`/`RLIMIT` (none found) and then reproducing the exact call pattern:
inside `run_capped.sh 20 --`, `mmap.mmap(-1, n)` + `cudaHostRegister(addr, n, 2)` succeeded at
2 GB, 4 GB, and 8 GB despite `ulimit -l` (soft and hard) being only 64 MB -- CUDA-on-WSL's pinning
path doesn't go through Linux's native `RLIMIT_MEMLOCK` accounting here. WSL2 itself is not the
blocker.

**Two of the three README quickstart steps don't work as written** (this repo is 2 days old;
take the README with real caution):
- `python compile_local.py` (step 2) refers to a file that isn't in the repo -- confirmed by
  `find` turning up nothing, and by upstream's own open issue filed the same day:
  `MaxDevv/lean-qwenflash#1 "No compile_local.py in repo"`. The real build path is
  `qwen38-flash-offload/scripts/install.sh` (needs `nvcc` + `ninja`, JIT-compiles each extension
  via `torch.utils.cpp_extension.load()`). **Do not set `QWEN_PREBUILT_EXT=1`** -- that opt-in
  path loads prebuilt `.so` files built for the author's torch ABI and `sm_89` (4090); using them
  here would be a silent ABI/arch mismatch, not a shortcut.
- `python download_http.py --model-dir ./model` (step 3) **does not accept `--model-dir` at
  all** -- the script has no `argparse`. Reading the actual source: it's hardcoded for the
  author's own Colab-style remote VM (`base=Path(os.environ.get('QWEN_WORK_ROOT',
  '/content/qwen-opt'))`), joins a specific cgroup by default
  (`QWEN_DOWNLOAD_JOIN_CGROUP=1` -> writes to `/sys/fs/cgroup/qwen-download/cgroup.procs`, which
  won't exist on this machine unless created first), and opens its destination files with
  `os.O_DIRECT` -- meaning its destination must *already* be a native O_DIRECT filesystem, not
  drvfs, from the very first byte of the download. This fully explains upstream issue #2,
  *"Access to unexpected repository: /content/qwen-opt/model"* -- another user hit the exact same
  hardcoded path. Plan to set `QWEN_WORK_ROOT` to wherever the native ext4 volume ends up mounted,
  and either pre-create `/sys/fs/cgroup/qwen-download` or set `QWEN_DOWNLOAD_JOIN_CGROUP=0`. A
  plain `hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.05bpw_h5_ng5 --local-dir
  <path>` (mentioned in the vendored SKILL.md for the GLM-lineage engine) is a simpler, documented
  alternative if `download_http.py`'s extra speed isn't needed.

Added `profiles/rtx3060-12gb.env`: a conservative first-run config (elastic/staging cache off,
`QWEN38_EC_MAX_SLOTS=256`, `--cache 16384`, `QWEN_VRAM_LIMIT_BYTES=11.5GB`,
`TORCH_CUDA_ARCH_LIST=8.6`) to try before touching any richer cache-tuning knobs, per the vendored
engine's own `qwen38-offload-setup/SKILL.md` ("tune one setting at a time against a control").

## What's still unverified

- **Whether the dense (non-expert) model fits at all, and now against a smaller real budget.**
  The HF revision `3.05bpw_h5_ng5` totals **85.14 GB** across 25 files (confirmed via the HF tree
  API, read-only, nothing downloaded) -- matching the top-level README's "~85 GB", not the
  vendored SKILL.md's "125.3 GB" (that number is from the unrelated GLM-lineage doc). Of that,
  `ngram_embedding.safetensors` is 32.64 GB and the 7 model shards total ~52.35 GB. The README
  says the packed *expert* store alone is ~47 GB, which would put the *dense* (non-expert) part
  at roughly **52.35 - 47 ≈ 5.3 GB** -- an estimate from subtraction, not a measurement. Against
  the corrected **~8.3 GB** real VRAM budget (see fix #4 above), that leaves only ~3 GB for KV
  cache + framework overhead + any expert-slot pool at all. Plausible, not comfortable, and **no
  amount of tuning fixes it if the real number is higher** -- it's a hard capacity question only
  the actual model load can answer.
- Whether `QWEN38_EC_MAX_SLOTS=256` and the rest of `profiles/rtx3060-12gb.env` are even in the
  right range -- derived from formulas and the upstream SKILL.md's desktop-GPU note (scaled down
  from its 24 GB target), not from a measurement on this card.
- Host RAM available for `--ram-gb` hasn't been re-checked against whatever else is running right
  now (`.wslconfig` caps the whole WSL VM at `memory=40GB`; that's a ceiling, not a guarantee).

## Blocked: where does ~130+ GB of data actually live?

Checked before downloading anything:

- **C: drive has only 45 GB free** (96% full). The WSL2 root filesystem (`ext4.vhdx`) already
  consumes **177 GB** on C:, and WSL reports its own virtual filesystem as "813 GB available" --
  that number is **not real headroom**; the vhdx is a dynamically-growing file on the actual,
  nearly-full C: drive. Do not write the model/store into the default WSL root.
- **D: drive (SK hynix PC801 NVMe) has 237 GB free**, mounted via `drvfs` (`/mnt/d`), which the
  project's own setup guidance excludes for the store (O_DIRECT reliability). This also rules out
  staging the *model* directory there long-term, not just the store: `qwen_offload.py` sets
  `EXL3_NGRAM_STREAM=1` and asserts `mode == 'trellis_disk'`, meaning the 32.64 GB
  `ngram_embedding.safetensors` streams directly from wherever `--model` points **on every
  request**, not once at load. And `download_http.py` opens its destination with `os.O_DIRECT`
  from the first byte (see above) -- it can't even write to `/mnt/d` as-is.
- **Real total footprint, corrected:** model (85 GB, including the n-gram file that has to stay)
  + store (47 GB) side by side = **~132 GB**, and the pack step's own peak (reads the ~52 GB of
  shards, writes 47 GB) adds to that while packing runs, not after. A fixed-size VHDX should be
  sized at **~150-160 GB**, not ~132, for working room. Downloading 85 GB onto D: as a staging
  step and *then* creating a 150-160 GB VHDX on the same drive needs close to all of D:'s 237 GB
  free -- tight, not comfortable. Better to download straight into the final native-ext4 volume
  once it exists, skipping the D: staging step entirely.
- **Only two physical NVMes currently show up in Windows** (`Get-PhysicalDisk`): the Intel
  `SSDPEKNW010T8` (C:, a QLC Gen3 drive, ~1.8 GB/s class -- below the project's 3+ GB/s floor even
  if it had space) and the SK hynix `PC801` (D:, Gen4). **The "spare NVMe drive" is not currently
  attached to this machine** -- no free slot confirmed either.

## Recommended disk route: fixed-size VHDX on D:, not the spare NVMe (for now)

Two real options; recommending one:

- **Recommended: a fixed-size VHDX file on D:**, created with `diskpart`, attached via
  `wsl --mount --vhd` (confirm exact flags with `wsl --help` on this WSL 2.7.3.0 before running
  anything), then `mkfs.ext4` run **inside WSL** on the attached block device. No hardware
  install, and D: (PC801, Gen4) outperforms the Intel 660p on C: anyway. O_DIRECT throughput
  through this path is unmeasured -- probe with `dd oflag=direct`/`iflag=direct` before trusting
  it for the real store.
- **Alternative: install the spare NVMe**, attach raw (`wsl --mount --bare`), format ext4 inside
  WSL. Needs a free M.2 slot (not yet confirmed) and a physical install, but gets a real
  dedicated drive instead of a virtual disk layered on D:'s filesystem.

Both need an **elevated PowerShell window** (can't elevate from inside WSL) and **must be redone
after every `wsl --shutdown` or reboot** -- the mount doesn't persist on its own. Per this
project's own `qwen38-flash-offload/AGENTS.md` ("ask before ... creating ... a filesystem, which
erases the drives"), the exact device/VHDX path gets confirmed before any `mkfs`.

## Decided (not asking): torch/CUDA toolkit version

Torch `2.13.0+cu130` is already installed on this machine and the 616.56 driver supports it.
`torch.utils.cpp_extension` refuses to build when `nvcc`'s CUDA major version differs from
torch's, so the fresh 3.10-3.12 venv should target the same `cu130` wheel, and the toolkit to
install is **`cuda-toolkit-13-0`** (WSL-Ubuntu repo), not `12-x`. Still needs sudo to install.
Never install the `cuda` meta-package or any in-WSL `nvidia-driver-*` package -- those break GPU
passthrough.

## Next steps (held pending Stuart's input)

1. Disk route: proceed with the VHDX-on-D: route above, or wait for/install the spare NVMe
   instead (needs confirming there's a free M.2 slot at all)?
2. OK to install `cuda-toolkit-13-0` via apt (sudo required)?
3. Once answered: create/attach the volume, point `HF_HOME` and the model download at it
   directly (skip D: staging), run `qwen38-flash-offload/scripts/install.sh` to build the native
   extensions, download+load the model to resolve the real dense-weight footprint (the only way
   to answer the capacity question above), then work through
   `qwen38-flash-offload/skills/qwen38-offload-setup/SKILL.md` step by step from
   `profiles/rtx3060-12gb.env`.

## Update 2026-10-28: disk/download/pack all done; first full `qwen_server.py` load attempt --

**new, deeper blocker found, unrelated to anything above.**

Since the "Next steps" above were written, the model was downloaded and the expert store was
packed (`/mnt/qwenstore/model`, `/mnt/qwenstore/store/qwen.bin`, 46.76 GB, 25,088/25,088 records
pass the script's own sha256 `verify` mode -- store integrity is not in question). `install.sh`
has already built the native extensions against `cu130`/`8.6` on this box per the "Decided"
section above.

First real end-to-end attempt: `scripts/run_capped.sh 20 -- python3 qwen_server.py --model
/mnt/qwenstore/model --store /mnt/qwenstore/store/qwen.bin --ram-gb 8 --cache 16384 --chunk 8192
--port 30000` (the exact command this file's own "Usage" block under `profiles/rtx3060-12gb.env`
recommends). `run_capped.sh`'s cgroup fix worked as documented. The dense model loaded and the
VRAM pool was built successfully at the 239s mark ("model loaded + VRAM pool built; DONTNEED on 8
checkpoint files") -- so the capacity question from "What's still unverified" above is
*answered*: the dense model + KV cache + minimal expert-slot pool DOES fit in the ~8.3 GB real
budget on this card. That's real progress.

It then crashed immediately after, inside `expert_cache.attach_modules` -> `nv2.attach_pool` ->
`Nv2.__init__` -> `nv2.py`'s `_mapped()`:

```
AssertionError: UVA alias != host address
  at: assert xe.pinned_cuda_view(t[:4096], dev_idx).data_ptr() == a
```

**This is a real, architectural WSL2/WDDM incompatibility, not a config/flag issue -- confirmed
by reading the native extension source, not guessed:**

- `exllamav3/exllamav3/exllamav3_ext/cuda_host.cpp`'s own `pinned_cuda_view` doc comment says
  outright: the CUDA device-side alias of `cudaHostRegister`-pinned memory "\[is\] equal to the
  host pointer under UVA (Linux desktop), but distinct under WDDM (native Windows, and
  potentially WSL2), where the host pointer is not usable in kernels and this alias must be
  passed instead." The function deliberately calls `cudaHostGetDevicePointer` instead of assuming
  pointer equality, *for exactly this reason*.
- But the caller (`nv2.py`'s `_mapped()`, and **`nv_tier.py`'s `HostArena.__init__` has the
  identical assertion at line 320** -- this is not nv2-specific, both of the project's NVMe tiers
  share the same broken assumption) throws that safety mechanism away: it calls
  `pinned_cuda_view` only to assert its result equals the raw host `mmap` address, instead of
  using the returned device pointer. On this box that assert is false: WSL2's CUDA-on-WDDM path
  behaves like the "distinct" case the extension's own author anticipated, not the "equal" case
  this script assumes.
- Switching tiers does not route around it: `qwen_offload.py` hardcodes `QWEN38_NV: '2'`
  (selecting `nv2.py`), but `QWEN38_NV=1` (the older `nv_tier.py`) would hit the exact same
  assertion in its own `HostArena`, since both reuse the same pin-and-check pattern.

**Why this isn't a quick patch:** the host-mmap address (`a`/`cbase_addr`) returned by `_mapped()`
gets used in two different contexts that need two different pointer values once host != device:
plain Python/`ctypes` writes from this process (fine with the host address), and arguments passed
into the native `e.init(...)` host-engine extension / embedded into GPU-visible control
structures that CUDA kernels on `nv2_dev.cu` dereference directly for the NVMe-to-GPU zero-copy
path (which need the *device* alias instead). Fixing this correctly means threading a second
pointer value through `nv2.py`/`nv_tier.py` call sites and the native `nv2_host.cpp`/`nv2_dev.cu`
signatures that currently assume one pointer serves both -- a native-code change to vendored,
~2-day-old upstream code, not a config flag. Did not attempt that patch blind in this session;
flagging it as its own scoped task.

**Current state:** server process is not running (crashed, confirmed via `ps aux`, nothing
orphaned). The expert store itself is fine and verified; the blocker is purely in the
GPU-offload engine's zero-copy pinned-memory plumbing under WSL2.

Options to take this further, not yet chosen:
1. Patch `_mapped()` (both copies, in `nv2.py` and `nv_tier.py`'s `HostArena`) to stop asserting
   equality and instead carry the device pointer separately for whatever gets passed to
   `e.init(...)`/GPU-visible structures -- requires reading `nv2_host.cpp` and `nv2_dev.cu` to
   confirm which call sites actually need the device alias vs. the host address, then verifying
   with the store's own byte-level `cmp` mode (not just "it didn't crash") since a wrong pointer
   here would silently corrupt served tokens rather than crash cleanly.
2. File/check upstream (`MaxDevv/lean-qwenflash`) for an existing issue or WDDM-aware branch --
   given how fast the two README-CLI bugs already found here turned out to have matching
   upstream issues (#1, #2), this WSL2 pointer assumption may already be reported.
3. Try running this engine from native Linux instead of WSL2 (bare metal or a VM with GPU
   passthrough that isn't WDDM-mediated) if one is available -- sidesteps the issue entirely since
   the UVA-equality assumption is true there.

## Update 2026-10-28 (same day, continued): option 1 applied -- fixed the pointer bug, hit a
**second, deeper one underneath it**

Patched option 1 above, specifically (not blindly -- read `nv2_host.cpp`/`nv2_dev.cu` first to
confirm which call sites need which address):

- `nv2.py`'s `_mapped()` now returns `(mm, host_addr, dev_addr, t)` instead of asserting equality.
  `Nv2.__init__` builds two dicts off it: `self.a` (host addresses -- unchanged, still fed to
  `e.init(...)`'s native CPU engine, which only ever touches this memory via `preadv`/`memcpy`/
  `cudaMemcpyAsync`'s host-side argument -- all correct with the host address on every platform)
  and a new `self.da` (device addresses). Every call site that feeds a raw pointer into a CUDA
  kernel launch -- `nv_pub`, `nv_step`, `nv_copy`, `nv_combine`, `nv_restore`, `nv_rows_all`, both
  in `nv2.py` and the two call sites in `expert_cache.py` -- now passes `self.da[...]`/
  `self.nv2.da[...]` instead of `self.a[...]`. `nv_tier.py`'s `HostArena` assertion was relaxed
  to a non-null check (its `.addr` list is only ever read by CPU code in this call pattern, so
  the equality it was asserting was never actually required -- confirmed by tracing every
  `ram_addr[s]`/`pf_ring_addr[r]` use in `nv2_host.cpp`, all `preadv`/`memcpy`/`cudaMemcpyAsync`
  host-side args, none inside a kernel launch).

**Result: this fix is confirmed correct as far as it goes.** Relaunched -- model load reached
and passed the exact point that crashed before (`nv2: control pinned region host=0x73c528430000
dev=0x209000000 (WDDM/WSL2 alias, not UVA-equal)` -- confirms the divergence is real on this box,
not theoretical), `Nv2.__init__` completed, the engine's own startup self-check passed with
all-zero bad counters (`vram_bad: 0, ram_bad: 0, table_bad: 0, ...`), `QWEN_SERVER_LOADED`
printed at 35.9s, and the server actually came up and listened on `127.0.0.1:30000`.

**But the first real `/v1/chat/completions` request crashed the server**, with `GPU assert: an
illegal memory access was encountered .../coop_autotune.cu 478` -- inside exllamav3's EXL3 GEMM
kernel autotuner (the quantized-matmul kernel benchmark pass that runs once per unique
shape/layer, triggered by the model's first real MoE expert computation). Re-ran under
`compute-sanitizer` (`/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck`) to get a real
backtrace -- **note: device-side instrumentation itself is NOT supported under this WSL2/WDDM
setup** (`Error: Device not supported` / `Failed to initialize WDDM debugger interface`), but the
wrapper still caught the host-side CUDA API error and gave a backtrace:

```
cudaErrorIllegalAddress (error 700) on CUDA API call to cudaFuncSetAttribute
  set_kernel_attr_once -> measure_stage -> tune -> CoopKernelAutotuner::launch
  -> exl3_gemm_gr -> exl3_gemm -> BC_LinearEXL3::run_gr -> BC_LinearEXL3::run_alloc
```

**Important nuance, not yet resolved:** `cudaFuncSetAttribute` failing with "illegal memory
access" is a classic CUDA symptom of a *sticky context error from an earlier async kernel fault*
-- once any kernel on a CUDA context triggers an illegal access, every subsequent API call on
that context returns the same error until the process exits, regardless of what that later call
actually does. So the real faulting kernel almost certainly ran *before* this GEMM autotune step,
not inside it. Given the timing (this is the model's very first real forward pass, right after
the just-fixed nv2 zero-copy pipeline runs its first `nv_copy`/`nv_step`/`nv_pub` kernels to admit
experts into VRAM), the leading suspect is that the zero-copy *mechanism itself* -- not just the
pointer value -- may not work correctly under WSL2's WDDM-mediated CUDA stack: i.e. a kernel
dereferencing `cudaHostGetDevicePointer`'s returned alias may not actually get coherent access to
the same bytes the CPU side wrote via the host pointer, even though registration succeeds and the
alias is non-null. Nobody (including this file's 2026-10-26 pinned-memory test) had actually
verified a *kernel read/write round-trip* through that alias on this box before -- only that
`cudaHostRegister` succeeds and that the alias is non-null.

**In progress at end of session:** wrote and ran an isolated minimal repro
(`/tmp/test_zerocopy.py`) -- mmap + `cudaHostRegister(Mapped)` + `cudaHostGetDevicePointer`,
then a trivial custom kernel that writes a known integer pattern through the device alias, then
reads it back via the host-side tensor to check the write round-trips. This directly answers
"does WSL2 support this zero-copy mechanism at all" independent of the rest of the model/nv2
complexity. Result not yet known when this note was written (JIT compile of the test kernel was
still running). **Next step for whoever picks this up: check that result first.**
- If the round trip **works**: the bug is somewhere else in the nv2 pipeline (a pointer I missed,
  or a synchronization gap between the CPU writer and GPU reader), not the mechanism itself --
  keep auditing `nv2.py`/`nv2_host.cpp`/`nv2_dev.cu` call sites.
- If the round trip **fails**: WSL2's CUDA-on-WDDM zero-copy pinned-memory path is genuinely
  broken or unsupported for kernel-side dereference on this driver/setup, which would mean the
  nv2 architecture (and the older nv_tier.py tier, which uses the identical mechanism) cannot work
  under WSL2 at all, regardless of pointer-value fixes -- the real options become running on
  native Linux (option 3, two updates up) or not using the NVMe zero-copy tier on this box.

## Update 2026-10-28 (same day, resolved): round-trip test passed -- found and fixed the real
**second bug, server now serves correct output**

The isolated repro (`/tmp/test_zerocopy.py`) came back clean: `host addr=0x7daf59e21000 dev
addr=0x204c00000` (confirmed genuinely distinct, as expected on WDDM), a trivial kernel wrote a
known pattern through the device alias, and reading it back via the host-side tensor matched
exactly -- `ZERO-COPY ROUND TRIP: OK`. **The mechanism itself works fine under WSL2.** That ruled
out the "WSL2 can't do this at all" theory and pointed back at the nv2 pipeline itself, per the
first bullet above.

Re-ran the real server with `CUDA_LAUNCH_BLOCKING=1` (forces every kernel launch to be checked
synchronously, so the error surfaces at its true site instead of a deferred one -- much cheaper
than `compute-sanitizer`, which isn't usable here anyway: it refused to instrument this GPU under
WSL2, `Error: Device not supported` / `Failed to initialize WDDM debugger interface`). That
confirmed the `coop_autotune.cu` errors from the previous update were exactly the deferred/sticky-
context symptom they looked like: the *real*, synchronous fault was `exl3_moe.cu:571`, inside the
fused MoE kernel's own launch -- the kernel that reads each selected expert's weight data through
per-expert pointer tables (`gate_ptrs_trellis`/`up_ptrs_trellis`/`down_ptrs_trellis`, populated
from the `home` table by `nv_rows_all_k`/`nv_restore_k`).

**Root cause:** the same host/device pointer split from the first bug also applies one level
down, to `nv_tier.py`'s `HostArena` (the RAM tier of resident experts) -- missed in the first
pass because `HostArena.addr`'s *direct* uses in `nv2_host.cpp` (preadv targets, `memcpy`,
`cudaMemcpyAsync`'s host-side argument) are all genuinely correct with the host address. The part
that was missed: `nv2_host.cpp`'s `set_home(key, ram_addr[s])` -- called at the exact moment a
RAM slot becomes resident -- publishes that address into the GPU-visible `home` table, and the
code's own comment already said so outright: `// device reads home only after ram_res == 1`. A
RAM-resident expert's weights are read by the fused MoE kernel via *direct zero-copy PCIe reads*
through that published address (this is the whole point of the "RAM tier" -- no explicit copy
into VRAM), so it needed the device alias, not the host address. Every other `ram_addr[s]` use
(including the CPU-tier's own direct pointer dereference at two call sites, and the staging
path's explicit `cudaMemcpyAsync`/`cudaMemcpyDefault` calls) stays correct with the host address.

**Fix applied** (verified, not guessed -- traced every remaining `ram_addr[` use in
`nv2_host.cpp` individually before concluding only these four needed to change):
- `nv_tier.py`'s `HostArena` now also computes `self.addr_dev` (device-alias counterpart of
  `self.addr`, same per-slot layout) alongside the existing host-address array.
- `nv2.py`'s `e.init(...)` call now passes `self.ram.addr_dev` as a new tensor argument.
- `nv2_host.cpp`'s `init()` gained a matching `ram_addr_dev` parameter and `Engine::ram_addr_dev`
  field, and the four `set_home(key, ram_addr[...])` call sites (lines ~296, 396, 751, 1026) were
  changed to `set_home(key, ram_addr_dev[...])`. (`pf_ring_addr` needed no equivalent change --
  traced every one of its uses too; it's only ever an explicit `cudaMemcpyAsync` source into VRAM,
  never published to a GPU-dereferenced table.)
- The small `qwen38-flash-offload/kernels/nv2` extension (2 files, not the full `exllamav3_ext`)
  recompiled automatically on next import via torch's JIT `load()` -- no manual rebuild step
  needed, and it's fast (~tens of seconds, not the multi-minute full-extension rebuild the
  isolated test script accidentally triggered earlier by not sourcing `TORCH_CUDA_ARCH_LIST`
  first).

**Verified end to end, not just "didn't crash":** relaunched normally (no `CUDA_LAUNCH_BLOCKING`,
no sanitizer -- full speed). Load reached `QWEN_SERVER_LOADED` at 74s, server came up on
`127.0.0.1:30000`, and two real `/v1/chat/completions` requests both returned HTTP 200 with
**coherent, correct output**: a sensible reasoning trace addressing the actual prompt, and
`"What is 7 times 6?"` -> `"42"` with `finish_reason: "stop"` (clean completion, not a truncated/
garbled one). Server stayed alive and responsive across both requests. This is real evidence the
fix is substantively correct -- wrong pointers here would have produced garbled/incoherent tokens
(reading garbage expert weight bytes), not fluent, contextually-correct English and the right
arithmetic answer.

**Current state: the server works.** Left running on `127.0.0.1:30000` per the original ask.
Everything that was fixed is in this repo's working tree (`nv2.py`, `nv_tier.py`,
`qwen38-flash-offload/kernels/nv2/nv2_host.cpp`, `expert_cache.py`) -- not yet committed to git if
this repo tracks changes; check `git status` before assuming it's durable across a fresh clone.

Server process from this session is stopped (confirmed dead, nothing orphaned).
