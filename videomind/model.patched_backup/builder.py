from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


def _resolve_base_model_path(model_path: str) -> str:
    model_dir = Path(model_path)
    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
            base_model_path = cfg.get("base_model_path", None)
            if base_model_path:
                return str(Path(base_model_path).expanduser().resolve())
        except Exception:
            pass
    return str(model_dir.resolve())


def _resolve_device(device: str | None) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def _load_generation_model(base_model_path: str, dtype: torch.dtype):
    common_kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if Qwen2VLForConditionalGeneration is not None:
        return Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path,
            **common_kwargs,
        )

    if AutoModelForVision2Seq is not None:
        return AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            **common_kwargs,
        )

    raise ImportError(
        "Neither Qwen2VLForConditionalGeneration nor AutoModelForVision2Seq "
        "is available in the current transformers version."
    )


def _load_extra_weights_if_any(model, model_path: str):
    model_dir = Path(model_path)

    extra_safetensors = model_dir / "pytorch_model.safetensors"
    if extra_safetensors.is_file() and safe_load_file is not None:
        state_dict = safe_load_file(str(extra_safetensors), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[builder] missing keys when loading extra weights: {len(missing)}")
        if unexpected:
            print(f"[builder] unexpected keys when loading extra weights: {len(unexpected)}")
        return model

    extra_bin = model_dir / "pytorch_model.bin"
    if extra_bin.is_file():
        state_dict = torch.load(str(extra_bin), map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[builder] missing keys when loading extra weights: {len(missing)}")
        if unexpected:
            print(f"[builder] unexpected keys when loading extra weights: {len(unexpected)}")
        return model

    return model


def build_model(model_path: str, device: str | None = "auto"):
    model_path = str(Path(model_path).expanduser().resolve())
    base_model_path = _resolve_base_model_path(model_path)
    device = _resolve_device(device)
    dtype = _resolve_dtype(device)

    print(f"Loading processor from {base_model_path}...")
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        do_resize=False,
        local_files_only=True,
        trust_remote_code=True,
    )

    print(f"Loading base generation model from {base_model_path}...")
    model = _load_generation_model(base_model_path, dtype=dtype)

    # If model_path is an adapter/checkpoint wrapper directory, try to load
    # additional non-base weights stored beside its config.
    if Path(model_path).resolve() != Path(base_model_path).resolve():
        model = _load_extra_weights_if_any(model, model_path)

    model = model.to(device)
    model.eval()
    return model, processor
