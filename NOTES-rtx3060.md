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

## 2026-10-28 session notes (WSL2 UVA finding + option 2 patch work)

- Mount: `/mnt/qwenstore` is ext4 on D:\wsl-disks\qwen-store.vhdx. Re-attach after wsl --shutdown:
  powershell.exe wsl.exe --mount --vhd D:\wsl-disks\qwen-store.vhdx --bare ; then
  /mnt/c/Windows/System32/wsl.exe -u root -e bash -c 'mount -o noatime LABEL=qwenstore /mnt/qwenstore' (root via wsl.exe works, no sudo password needed).
- CUDA 13.0 toolkit installed; native exts built in qwen38-flash-offload/build/*. Build/run needs
  CUDA_HOME=/usr/local/cuda-13.0 and PATH including .venv/bin (ninja). First server start compiles ~190 exllamav3 kernels (~20 min), now cached.
- Model (80G) + store (44G, qwen.bin) done. Load = 93 s, VRAM ~7GB transient.
- Launch: set -a; . profiles/rtx3060-12gb.env; set +a; scripts/run_capped.sh 26 -- python3 qwen_server.py --model /mnt/qwenstore/model --store /mnt/qwenstore/store/qwen.bin --ram-gb 12 --cache 16384 --chunk 4096 --port 30000
- FAILURE on WSL2: AssertionError "UVA alias != host address" (nv2.py:75 _mapped; nv_tier.py:320 HostArena). cudaHostRegister(Mapped) succeeds but
  exllamav3 pinned_cuda_view returns a device ptr != host ptr (host 0x7b40.. -> dev 0x204c00000). WDDM/WSL2 has no identity UVA. Verified with /tmp/uva_test.py.
- Patch plan (option 2, Stuart chose): host addrs (nv2.py cbase_addr/self.a[*], ram.addr, pf.addr) are passed to device kernels (nv2_dev.cu) and host engine (nv2_host.cpp).
  Need a separate device-address for each pinned region (cudaHostGetDevicePointer == pinned_cuda_view(t).data_ptr()) while host threads keep host addr.


### Patch scoping (option 2) - 2026-10-28
Root: qwen38-flash-offload/. Device kernels dereference HOST pointers directly (assume UVA identity). Needed split: host addr (CPU threads, memcpy, O_DIRECT reads) vs device addr (kernels).
Sites:
 - nv2.py _mapped(): drop the identity assert; return (mm, host_addr, tensor, dev_addr=pinned_cuda_view(t).data_ptr()) -> delta = dev - host per region.
 - nv2.py L146-147: self.a[k]=cbase_addr+off -> keep host dict self.a (for e.init host engine) AND new self.ad (device addrs) used for ALL kernel launches: nv_rows_all, nv_step, nv_copy(home,res), nv_restore (expert_cache.py L545), nv_combine, etc. (lines ~210,263,285,360 + expert_cache.py).
 - nv_tier.py HostArena (L305-325): per-chunk dev base; expose .addr (host) and .daddr (device) arrays. Remove identity assert.
 - kernels/nv2/nv2_host.cpp: set_home(key,a) writes into 'home' table which DEVICE reads (nv2_dev.cu home_all[key*3..]) -> must store DEVICE addr of slot (host ram_addr[s] + per-chunk delta). Lines 234,289,389,744,1016 call set_home(ram_addr[s]); engine init needs a parallel ram_daddr vector.
 - nv2_dev.cu nv_copy_k L410: s_wb = ram_addr[jobx[j]] (device-side tensor of RAM slot addrs, passed from python) -> must be device addrs. ram_addr_dev in nv2.py ~L172 / nv_copy arg.
 - nv2_host.cpp host-side uses of ram_addr (memcpy L750, cudaMemcpyAsync L378/1008, rd.submit reads L297) stay HOST addr. cudaMemcpyAsync with host-registered ptr is fine.
 - Region layout structs (Req/Rep/etc in control region) are accessed by host threads via host ptr and kernels via device ptr -> fine once two bases passed.
Uncertain: whether mapped zero-copy reads from kernels perform OK on WDDM (test needed).


### Patch plan v2 (exact) 2026-10-28
- nv2.py: _mapped returns dev base too (pinned_cuda_view(t).data_ptr()); new self.ad = {k: dev_base+off[k]}; keep self.a (host) ONLY for e.init (L194-195). Replace self.a[...] with self.ad[...] in kernel launches: L210,263,285,358-366,415,436 and expert_cache.py L545,562 (nv2.a['home'] -> nv2.ad['home']).
- nv_tier.HostArena: add self.dbases (dev per chunk) and self.daddr array; remove identity assert.
- nv2_host.cpp: init() gets extra at::Tensor ram_daddr; Engine.ram_daddr vector; set_home(key, ram_addr[s]) at ~L289,389,744,1016 -> ram_daddr[s] (home table is read by DEVICE). set_home(vring_addr) L325 unchanged (already device ptrs). Host memcpy/reads/cudaMemcpyAsync keep host addrs.
- nv2.py L196 init call: pass torch.tensor(self.ram.daddr.tolist()) as new arg (position after pf_ring).
- Backups: *.orig next to each patched file. Rebuild nv2 ext: rm build/nv2/*.so, rerun install step (nv2 only).


### Patch status 2026-10-28 ~08:55
APPLIED (backups *.orig in qwen38/ and kernels/nv2/): Python (nv2.py: _mapped returns dev base, self.ad dict used for all kernel launches; expert_cache.py nv2.ad; nv_tier.HostArena dbases/daddr, asserts removed) and C++ nv2_host.cpp (ram_daddr vector, init() takes new ram_daddr tensor AFTER pf_ring, 4 set_home calls use ram_daddr).
STILL TO CHECK: (1) nv2.py ram_addr_dev/vring: nv_copy_k uses ram_addr[jobx[j]] = vring addrs (device ptrs already, fine). (2) device kernels read 'ph' etc fine. (3) any other kernel that gets HOST addrs of ram slots: expert_cache.py nv_restore/staging (st_ram_src in host cpp L829 uses ram_addr for H2D cudaMemcpyAsync - host ptr OK). (4) nv_tier.py self.ring.addr tolist L636 passed to kernel? check. (5) nv_tier.py L683/nv2.py L587 debug compares home[k,0] != ram.addr -> use daddr (debug only).
NEXT: rebuild nv2 ext (rm build/nv2/*.so; rerun install.sh with env), relaunch server per launch cmd in notes, watch /tmp/qwen-server.log.


### RESULT 2026-10-28 08:56: UVA patch WORKS on WSL2/RTX 3060
- nv2 engine up, server on :30000, coherent output on 3 prompts. Load ~110 s (cached kernels). VRAM steady ~8.0 GB of 12 GB. RAM tier 6990 slots (12 GiB).
- Speed: 200 tokens in 48.6 s = 4.1 tok/s (incl. 60-token prefill), temperature 0. NVMe store read ~1.5 GB/s from the ext4 VHDX.
- Caveat: only short prompts tested; long-context, concurrency and sustained runs untested. Output includes reasoning_content field.


### Tuning results 2026-10-28 ~09:12 (warm, 3 prompts x150 tok, temp 0)
- QWEN38_EC_MAX_SLOTS=256 (profile default): ~4.7 tok/s, VRAM ~8.0GB
- =1000: 6.17/6.80/6.31 (avg ~6.4), VRAM 9.3GB
- =1600: 6.82/7.28/7.18 (avg ~7.1), VRAM 10.4GB (cap 11.5GB). Server running PID 47766 at 1600. ~+50% vs default.
- Bench script /tmp/bench.sh (may be lost on reboot). Headroom ~1GB under cap; 2000+ risks OOM with desktop (dwm/Discord) VRAM drift. Remaining bottleneck: ~11% of picks go to NVMe (1.5GB/s VHDX).
- TODO: set QWEN38_EC_MAX_SLOTS=1600 in profiles/rtx3060-12gb.env, commit to rtx3060-uva-patch-buddy.


### Context + speed push 2026-10-28 ~09:25
- cache 16K@1600 slots: ~7.1 tok/s, 10.4GB | 32K@1600: 6.85/7.51/6.60 (avg 7.0), 10.5GB | 64K@1600: DIES at load (VRAM 11.57GB > 11.5GB watchdog) | 64K@1200: 6.65/7.26/6.71 (avg 6.9), 10.2GB.
- CPU expert tier (QWEN_CPU_TIER=1, env hook added in qwen_server.py, default off): crashes in nv2._selftest, HostArena.view() chunk offset bug (offset 941875200 > buffer 615628800). Not fixed.
- 20-30 tok/s NOT reached. Upstream gets 20+ with 16GB VRAM / 8GB+ RAM-cache on a 4090/3080L; here ~11% of picks hit NVMe (1.5GB/s VHDX), VRAM holds only ~1200-1600 slots of 24576.
- Run: /tmp/cycle.sh style: QWEN38_EC_MAX_SLOTS=1200 ... --cache 65536.
