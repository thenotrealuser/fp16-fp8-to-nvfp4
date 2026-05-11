import argparse
import gc
import json
import os
import sys
import time
from collections import OrderedDict, Counter
from pathlib import Path


def log(msg=""):
    print(msg, flush=True)


BLACKLISTS = {
    # Profile matching the public Kitchen node for Z-Image-Turbo.
    "Z-Image-Turbo": [
        "cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer",
    ],
    # Safer but bigger. Useful for FP8 source or finetunes where attention degradation is ugly.
    "Z-Image-Turbo-Conservative": [
        "attention", "adaLN_modulation", "norm", "cap_embedder", "x_embedder", "noise_refiner",
        "context_refiner", "t_embedder", "final_layer",
    ],
    "Z-Image-Base": [
        "attention", "adaLN_modulation", "norm", "final_layer", "cap_embedder", "x_embedder",
        "noise_refiner", "context_refiner", "t_embedder",
    ],
    "Flux.1-dev": [
        "bias", "txt_attn", "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer",
        "class_embedding", "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt",
    ],
    "Flux.1-Fill": [
        "bias", "txt_attn", "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer",
        "class_embedding", "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt",
    ],
    "Flux.2-dev": [
        "bias", "txt_attn", "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer",
        "class_embedding", "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt",
    ],
    "Flux.2-Klein-9b": [
        "bias", "txt_attn", "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer",
        "class_embedding", "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt",
    ],
    "Qwen-Image-Edit-2511": ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out"],
    "Qwen-Image-2512": ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out", "img_mod.1"],
    "Wan2.2-i2v-high-low": ["text_embedding", "time_embedding", "time_projection", "head"],
    "LTX-2-19b-dev-or-distilled": [
        "vae.", "vocoder.", "connector", "proj_out", "norm", "bias", "scale", "embedder",
        "patchify", "table", "transformer_blocks.0.", "transformer_blocks.43.",
        "transformer_blocks.44.", "transformer_blocks.45.", "transformer_blocks.46.",
        "transformer_blocks.47.", "projection", "adaln_single",
    ],
}

FP8_LAYERS = {
    "Qwen-Image-2512": ["txt_mlp", "txt_mod"],
}


def import_deps():
    try:
        import torch
        import safetensors
        import safetensors.torch
        from safetensors import safe_open
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import TensorCoreNVFP4Layout
        return torch, safetensors, safe_open, ck, TensorCoreNVFP4Layout
    except Exception as e:
        log("ERROR: Failed to import required modules.")
        log(f"{type(e).__name__}: {e}")
        log("Fix: run install_venv.bat or install_deps_only.bat")
        sys.exit(2)


def base_meta_key(k: str) -> str:
    base = k.replace(".weight", "")
    if "model.diffusion_model." in base:
        return base.split("model.diffusion_model.", 1)[-1]
    return base


def is_quantizable_weight(k, v):
    return k.endswith(".weight") and getattr(v, "ndim", 0) == 2


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def is_fp8_dtype_name(name):
    return "float8" in name.lower() or "fp8" in name.lower()


def gpu_line():
    try:
        import subprocess
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL, timeout=3).strip().splitlines()[0]
        return out
    except Exception:
        return "n/a"


def scan_file(path, model_type):
    torch, safetensors, safe_open, ck, TensorCoreNVFP4Layout = import_deps()
    path = Path(path)
    blacklist = BLACKLISTS.get(model_type, BLACKLISTS["Z-Image-Turbo"])
    dtype_counts = Counter()
    quantizable = 0
    blacklisted_quantizable = 0
    total_tensors = 0
    total_bytes = path.stat().st_size if path.exists() else 0
    examples = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
        for k in keys:
            total_tensors += 1
            v = f.get_tensor(k)
            dn = dtype_name(v.dtype)
            dtype_counts[dn] += 1
            if is_quantizable_weight(k, v):
                if any(name in k for name in blacklist):
                    blacklisted_quantizable += 1
                else:
                    quantizable += 1
                    if len(examples) < 10:
                        examples.append(k)
            del v
    return {
        "path": str(path),
        "size_gb": round(total_bytes / (1024**3), 2),
        "model_type": model_type,
        "total_tensors": total_tensors,
        "quantizable_2d_weights": quantizable,
        "blacklisted_2d_weights": blacklisted_quantizable,
        "dtype_counts": dict(dtype_counts),
        "metadata_keys": list(meta.keys()),
        "examples": examples,
        "fp8_detected": any(is_fp8_dtype_name(k) for k in dtype_counts),
    }


def do_scan(args):
    info = scan_file(args.input, args.model_type)
    log("=== Dry scan ===")
    log(json.dumps(info, indent=2, ensure_ascii=False))
    if info["fp8_detected"] and not args.allow_fp8:
        log("\nWARNING: FP8 source detected. Conversion is blocked unless --allow-fp8 is used.")
        log("Recommended source is BF16/FP16 original. FP8 -> NVFP4 can degrade quality.")
        return 8
    return 0


def convert(args):
    torch, safetensors, safe_open, ck, TensorCoreNVFP4Layout = import_deps()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".safetensors":
        output_path = output_path.with_suffix(".safetensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        log("ERROR: CUDA selected but torch.cuda.is_available() is False.")
        return 3

    blacklist = BLACKLISTS.get(args.model_type, BLACKLISTS["Z-Image-Turbo"])
    fp8_layers = FP8_LAYERS.get(args.model_type, [])

    log("=== Z-Image NVFP4 Kitchen converter ===")
    log(f"Input: {input_path}")
    log(f"Output: {output_path}")
    log(f"Model type/profile: {args.model_type}")
    log(f"Device: {args.device}")
    log(f"Allow FP8 source: {args.allow_fp8}")
    log(f"Input size: {round(input_path.stat().st_size / (1024**3), 2)} GB")
    log(f"GPU: {gpu_line()}")
    log("")

    scan = scan_file(input_path, args.model_type)
    log("Scan summary:")
    log(json.dumps({k: scan[k] for k in ["total_tensors", "quantizable_2d_weights", "blacklisted_2d_weights", "dtype_counts", "fp8_detected"]}, indent=2))
    if scan["fp8_detected"] and not args.allow_fp8:
        log("ERROR: FP8 source detected, but --allow-fp8 was not passed.")
        log("Use BF16/FP16 source for best quality, or pass --allow-fp8 to force conversion.")
        return 8

    new_sd = {}
    quant_map = {
        "format_version": "1.0",
        "layers": {},
        "source_model_type": args.model_type,
        "source_file": input_path.name,
        "source_dtype_counts": scan["dtype_counts"],
        "warning": "Generated by an unofficial standalone wrapper using comfy-kitchen TensorCoreNVFP4Layout.",
    }

    source_metadata = {}
    start = time.time()
    converted = 0
    kept = 0
    failed = 0

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        source_metadata = f.metadata() or {}
        keys = list(f.keys())
        total = len(keys)
        for i, k in enumerate(keys, 1):
            try:
                v = f.get_tensor(k)
                dn = dtype_name(v.dtype)
                pct = i * 100.0 / max(1, total)
                if i == 1 or i % args.progress_every == 0 or i == total:
                    log(f"PROGRESS|{i}|{total}|{pct:.2f}|converted={converted}|kept={kept}|failed={failed}|key={k}|gpu={gpu_line()}")

                if any(name in k for name in blacklist):
                    # Keep sensitive tensors in BF16.
                    if torch.is_floating_point(v):
                        new_sd[k] = v.to(dtype=torch.bfloat16).cpu()
                    else:
                        new_sd[k] = v.cpu()
                    kept += 1
                    del v
                    continue

                if is_quantizable_weight(k, v):
                    base_file = k.replace(".weight", "")
                    base_meta = base_meta_key(k)
                    v_tensor = v.to(device=args.device, dtype=torch.bfloat16)
                    del v

                    # Optional FP8 fallback path for named layers. Mostly retained for profile parity.
                    if fp8_layers and any(name in k for name in fp8_layers):
                        weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                        weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                        new_sd[k] = weight_quantized.cpu()
                        new_sd[f"{base_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                        quant_map["layers"][base_meta] = {"format": "float8_e4m3fn"}
                        converted += 1
                        del v_tensor
                        if args.device == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    try:
                        qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                        tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                        for suffix, tensor in tensors.items():
                            new_sd[f"{base_file}.weight{suffix}"] = tensor.cpu()
                        quant_map["layers"][base_meta] = {"format": "nvfp4"}
                        converted += 1
                    except Exception as e:
                        log(f"WARN: NVFP4 failed for {k}: {type(e).__name__}: {e}. Keeping BF16.")
                        new_sd[k] = v_tensor.to(dtype=torch.bfloat16).cpu()
                        failed += 1
                    finally:
                        del v_tensor
                        if args.device == "cuda":
                            torch.cuda.empty_cache()
                    continue

                # Non-2D tensors: keep BF16 if floating, otherwise keep as-is.
                if torch.is_floating_point(v):
                    new_sd[k] = v.to(dtype=torch.bfloat16).cpu()
                else:
                    new_sd[k] = v.cpu()
                kept += 1
                del v

            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"ERROR processing {k}: {type(e).__name__}: {e}")
                failed += 1
                try:
                    del v
                except Exception:
                    pass
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                if not args.continue_on_error:
                    return 5

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = "Z-Image NVFP4 Kitchen GUI"
    final_metadata["converter_basis"] = "comfy-kitchen TensorCoreNVFP4Layout"
    final_metadata["model_type"] = args.model_type
    final_metadata["source_file"] = input_path.name
    # Preserve useful existing metadata where possible without overwriting our quantization map.
    for mk, mv in source_metadata.items():
        if mk not in final_metadata and isinstance(mv, str):
            final_metadata[mk] = mv

    log("")
    log("Saving safetensors...")
    safetensors.torch.save_file(new_sd, str(output_path), metadata=final_metadata)
    size_gb = output_path.stat().st_size / (1024**3)
    elapsed = time.time() - start
    log("DONE")
    log(f"Output size: {size_gb:.2f} GB")
    log(f"Converted layers: {converted}")
    log(f"Kept tensors: {kept}")
    log(f"Failed layers kept BF16: {failed}")
    log(f"Elapsed: {elapsed:.1f}s")
    log(f"Output: {output_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Standalone NVFP4 converter using comfy-kitchen TensorCoreNVFP4Layout")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-type", default="Z-Image-Turbo", choices=list(BLACKLISTS.keys()))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--allow-fp8", action="store_true", help="Allow FP8 source files. BF16/FP16 is recommended.")
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--progress-every", type=int, default=5)
    args = p.parse_args()

    if not Path(args.input).exists():
        log(f"ERROR: input not found: {args.input}")
        return 1

    if args.scan_only:
        return do_scan(args)
    return convert(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise SystemExit(130)
