"""Qwen3.8 EXL3 adapter for the GLM disk/RAM/VRAM expert engine."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load(model_dir, store, chunk=2048, cache_tokens=16384, ram_gb=6.0, cpu=False,
         mtp=0, embed_disk=False, cache_quant=None):
    import torch
    model_dir=Path(model_dir).resolve(); store=Path(store).resolve()
    sys.path.insert(0, str(ROOT / 'qwen38-flash-offload' / 'qwen38'))
    env = {
        'QWEN38_NV': '2', 'QWEN38_EC': '1', 'QWEN38_NV_STORE': str(store),
        'QWEN38_NV_RAM_GB': str(ram_gb), 'QWEN38_NV_CPU': str(int(cpu)),
        'QWEN38_RCS_GB': '0.25',
        'QWEN38_NV_EXCL': '1', 'QWEN38_NV_PREFETCH': '0',
        'QWEN38_NV_THREADS': '12', 'QWEN38_NV_READER_CPUS': '2-11',
        'QWEN38_NV_CTL_CPU': '1', 'QWEN38_NV_CPU_CPUS': '2-7',
        'QWEN38_NV_PIECE_KB': '256', 'QWEN38_NV_PF_RING': '512',
        'QWEN38_NV_PF_QD': '24', 'QWEN38_NV_VRING': '24',
        'QWEN38_EC_STAGE_GB': '1.1', 'QWEN38_EC_STAGE_MIN': '512',
        'QWEN38_EC_ELASTIC_GB': '6', 'QWEN38_EC_RESERVE_GB': '1.7',
        'QWEN38_NV_COPY_GRID': '24', 'QWEN38_EC_COPY_GRID': '24',
        'EXL3_NGRAM_STREAM': '1', 'QWEN_VRAM_LIMIT_BYTES': '16000000000',
        'MAX_JOBS': '4',
    }
    for k,v in env.items(): os.environ.setdefault(k,v)
    torch.set_num_threads(1)
    torch.cuda.init()
    # TORCH_CUDA_ARCH_LIST defaulted to '8.9' (Ada / RTX 4090), the card the upstream README was
    # benchmarked on. On any other architecture (e.g. sm_86 / Ampere, RTX 3060/3090) the compiled
    # kernels would be missing for this device and fail at first launch ("no kernel image
    # available"). Derive it from the actual GPU instead of assuming 4090-class hardware.
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    _, total = torch.cuda.mem_get_info()
    # Original hardcoded a 15 GB numerator, implicitly assuming a >=16 GB card. On a smaller or
    # desktop-shared (WDDM) card that fraction can exceed 1.0, which torch rejects outright
    # (ValueError) before anything else runs. Derive the budget from what's actually free right
    # now (so it accounts for VRAM already held by the desktop compositor / other GPU clients),
    # minus headroom for the CUDA context, allocator fragmentation, and fluctuation in whatever
    # else is sharing the GPU. Clamp into (0, 1] so this never raises regardless of card size.
    free, _ = torch.cuda.mem_get_info()
    headroom = max(1_500_000_000, int(total * 0.12))
    budget = max(0, min(free, total) - headroom)
    fraction = max(0.05, min(0.97, budget / total))
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f'QWEN_VRAM_BUDGET total={total/1e9:.2f}GB free={free/1e9:.2f}GB '
          f'headroom={headroom/1e9:.2f}GB fraction={fraction:.3f} '
          f"arch={os.environ['TORCH_CUDA_ARCH_LIST']}", flush=True)
    if os.environ.get('QWEN_EXL3_SOURCE')=='1':
        sys.path.insert(0,str(ROOT/'exllamav3'))
    from exllamav3 import model_init
    import argparse, nv_tier, expert_cache, nv2
    if os.environ.get('QWEN_PREBUILT_EXT') == '1':
        # Explicit opt-in for the recovered binaries built with this same Torch/CUDA ABI.
        import importlib.util
        for module, directory, name in ((nv_tier,'nv','qwen38_nv'),
                                        (expert_cache,'ec','qwen38_ec'), (nv2,'nv2','qwen38_nv2')):
            path=ROOT/'qwen38-flash-offload'/'build'/directory/(name+'.so')
            spec=importlib.util.spec_from_file_location(name,path)
            native=importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
            module._EXT=native
    if mtp: os.environ['QWEN_INCLUDE_MTP']='1'
    nv_tier.install(); expert_cache.install()
    ap=argparse.ArgumentParser(); model_init.add_args(ap,cache=True,add_draft_model_args=bool(mtp))
    argv=['-m',str(model_dir),'-cs',str(cache_tokens),'-rcs','0.25',
          '-chunk_size',str(chunk),'-ambs','1','-nw']
    if mtp: argv+=['-mtp','-ndt',str(mtp)]
    if embed_disk: argv+=['-embd']
    if cache_quant is not None: argv+=['-cq',str(cache_quant)]
    args=ap.parse_args(argv)
    loaded=model_init.init(args)
    if mtp:
        model,config,cache,tok,draft_model,_draft_config,draft_cache=loaded
        mods=expert_cache.find_moe(model.modules)+expert_cache.find_moe(draft_model.modules)
        pools=expert_cache.attach_modules(mods)
        expert_cache._hook_prefill(type(model)); expert_cache._hook_prefill(type(draft_model))
    else:
        model,config,cache,tok=loaded; draft_model=draft_cache=None
        pools=expert_cache.attach(model)
    assert config.arch_string == 'Qwen4ExpForConditionalGeneration'
    nv2.finish(model)
    from exllamav3.modules.ngram_embedding import NGramEmbedding
    def walk(mods):
        for m in mods:
            yield m
            yield from walk(getattr(m,'modules',[]) or [])
    tables=[m for m in walk(model.modules) if isinstance(m,NGramEmbedding)]
    assert tables and all(m.mode == 'trellis_disk' for m in tables), 'All n-gram tables must stream from disk'
    print('NGRAM_MODES',[(m.key,m.mode,m.num_rows) for m in tables],flush=True)
    return model,config,cache,tok,pools,tables,draft_model,draft_cache
