#!/usr/bin/env python3
"""RAVEL-VB players + ball detection on ONNX Runtime (CUDA on the GPU image).

A drop-in replacement for ``inference_player_ball_openvino.py``: the same
arguments, the same decoding, hysteresis, ``--resume`` and the same .json/.jsonl
output -- only the runtime differs. The model is a RAVEL-VB ``.onnx`` release
with its sidecar ``.json`` (same tensor names as the OpenVINO release):

    uv run python src/inference_player_ball_onnx.py video.mp4 \\
        --model models/RAVEL-VB-011-9f.onnx --device GPU --output ../uploads/mix/

``--device GPU`` runs on ``CUDAExecutionProvider``, ``CPU`` on the CPU, ``AUTO``
on CUDA when it is available. A GPU request that onnxruntime cannot satisfy is
an error, not a silent fallback to the CPU.

Environment (see ``Dockerfile.gpu``):

    ORT_CUDA_DEVICE_ID   CUDA device index, default 0
    ORT_CUDA_USE_TF32    1 lets convolutions run in TF32. Default 0: with TF32 on
                         an RTX 50xx one box in ~300 moves by 0.03 against the
                         CPU/OpenVINO result, for a ~10% speed-up
    ORT_CUDNN_CONV_ALGO  cuDNN algorithm search: EXHAUSTIVE (default), HEURISTIC
                         or DEFAULT
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

try:
    from .inference_player_ball_openvino import RavelVBOpenVINO, main
except ImportError:
    from inference_player_ball_openvino import RavelVBOpenVINO, main

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "RAVEL-VB-011-9f.onnx"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def _preload_cuda_libs() -> None:
    """Expose the CUDA/cuDNN libraries of an ``onnxruntime-gpu`` install.

    Same as in ``inference_onnx_seq_gray_v2.py``: the nvidia-* wheels put them
    into ``site-packages/nvidia/<pkg>/lib``, where the CUDA provider does not
    look by itself; without the preload it fails on ``libcublasLt.so.13``.
    On a CPU-only install there is nothing to load.
    """
    import site

    libs = [
        str(lib_dir)
        for site_dir in {*site.getsitepackages(), site.getusersitepackages()}
        for lib_dir in sorted(Path(site_dir).glob("nvidia/*/lib"))
        if lib_dir.is_dir()
    ]
    if libs:
        os.environ["LD_LIBRARY_PATH"] = ":".join([*libs, os.environ.get("LD_LIBRARY_PATH", "")])
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()


_preload_cuda_libs()


def _providers(device: str) -> list:
    available = ort.get_available_providers()
    if device == "CPU":
        return [CPU]
    if CUDA not in available:
        if device == "AUTO":
            return [CPU]
        raise RuntimeError(
            f"--device GPU needs onnxruntime-gpu with {CUDA}; available: {available}"
        )
    cuda_options = {
        "device_id": int(os.getenv("ORT_CUDA_DEVICE_ID", "0")),
        "use_tf32": int(os.getenv("ORT_CUDA_USE_TF32", "0")),
        "cudnn_conv_algo_search": os.getenv("ORT_CUDNN_CONV_ALGO", "EXHAUSTIVE"),
    }
    return [(CUDA, cuda_options), CPU]


class RavelVBOnnx(RavelVBOpenVINO):
    """``RavelVBOpenVINO`` with ONNX Runtime underneath.

    The metadata checks, ``preprocess``, the ball priors and the
    ``infer`` contract (a dict of output tensors and the seconds spent) are the
    OpenVINO ones; only the session is different. ``performance_hint`` is an
    OpenVINO setting and is ignored here.
    """

    def __init__(
        self,
        model_path: Path,
        num_threads: int,
        performance_hint: str,
        device: str,
    ) -> None:
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"model metadata not found: {metadata_path}")
        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.config = self.metadata["config"]
        required_config = {"image_width", "image_height", "clip_length", "input_channels"}
        missing_config = sorted(required_config - self.config.keys())
        if missing_config:
            raise ValueError(f"model metadata is missing config fields: {missing_config}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=_providers(device)
        )
        active = self.session.get_providers()
        if device == "GPU" and active[0] != CUDA:
            raise RuntimeError(f"CUDA provider did not start, onnxruntime uses {active}")
        self.device = "CUDA" if active[0] == CUDA else "CPU"

        self.input_name = self.metadata.get("input_name", "frames")
        session_inputs = [item.name for item in self.session.get_inputs()]
        if self.input_name not in session_inputs:
            raise ValueError(f"model has no input {self.input_name!r}: {session_inputs}")
        output_names = self.metadata.get("output_names")
        if not isinstance(output_names, list):
            raise ValueError("model metadata must contain an output_names list")
        missing_outputs = sorted({"logits", "boxes"} - set(output_names))
        if missing_outputs:
            raise ValueError(f"ONNX model is missing required outputs: {missing_outputs}")
        session_outputs = {item.name for item in self.session.get_outputs()}
        absent = sorted(set(output_names) - session_outputs)
        if absent:
            raise ValueError(f"metadata names outputs the model does not have: {absent}")
        # The same rule as the OpenVINO runner: a ball release must carry its priors.
        self.has_ball = "ball_grid" in output_names
        if self.has_ball:
            ball_missing = sorted({"ball_width_prior", "ball_height_prior"} - self.config.keys())
            if ball_missing:
                raise ValueError(f"ball release is missing config fields: {ball_missing}")
            self.ball_width = float(self.config["ball_width_prior"])
            self.ball_height = float(self.config["ball_height_prior"])
        else:
            self.ball_width = self.ball_height = 0.0
        self.output_names = list(output_names)

    def infer(self, clip: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        started = time.perf_counter()
        result = self.session.run(
            self.output_names, {self.input_name: np.ascontiguousarray(clip, dtype=np.float32)}
        )
        elapsed = time.perf_counter() - started
        return dict(zip(self.output_names, result)), elapsed


if __name__ == "__main__":
    main(RavelVBOnnx, DEFAULT_MODEL, default_device="GPU")
