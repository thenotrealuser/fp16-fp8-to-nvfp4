import importlib
import platform
import sys
import subprocess

print("=== Z-Image NVFP4 Kitchen env check ===", flush=True)
print("Python:", sys.version, flush=True)
print("Executable:", sys.executable, flush=True)
print("Platform:", platform.platform(), flush=True)
print("sys.path site-ish:", [p for p in sys.path if 'site-packages' in p or p.endswith('.venv')], flush=True)
print()

mods = [
    "torch",
    "safetensors",
    "comfy_kitchen",
    "psutil",
]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"Module {m}: OK {getattr(mod, '__file__', None)}", flush=True)
    except Exception as e:
        print(f"Module {m}: MISSING/ERROR {type(e).__name__}: {e}", flush=True)

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
    print("TensorCoreNVFP4Layout: OK", flush=True)
except Exception as e:
    print(f"TensorCoreNVFP4Layout: ERROR {type(e).__name__}: {e}", flush=True)

print()
try:
    import torch
    print("Torch version:", torch.__version__, flush=True)
    print("Torch CUDA:", torch.version.cuda, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("Compute capability:", ".".join(map(str, torch.cuda.get_device_capability(0))), flush=True)
        props = torch.cuda.get_device_properties(0)
        print("VRAM GB:", round(props.total_memory / (1024**3), 2), flush=True)
except Exception as e:
    print("Torch check failed:", type(e).__name__, e, flush=True)

print()
try:
    out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
    print("nvidia-smi:")
    print(out, flush=True)
except Exception as e:
    print("nvidia-smi unavailable:", type(e).__name__, e, flush=True)
