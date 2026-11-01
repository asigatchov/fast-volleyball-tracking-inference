#!/usr/bin/env python3
"""Export the vb-action clip classifier (Serve / Receive / Set / Attack / NoAction) to OpenVINO.

Needs torch, so it runs in the vb-action environment, from the vb-action repo root:

    PYTHONPATH=. .venv/bin/python /path/to/scripts/export_action_classifier.py \
        models/action_clf_r2plus1d18_pm6_noaction.pt /path/to/ov/action_clf_r2plus1d18_pm6_noaction

The exported graph includes the preprocessing of infer_action_clips.py (Kinetics
normalization and the centre view), so it takes the stored crops as they are:
``[B, T, S, S, 3]`` RGB in 0..255 and returns ``[B, classes]`` probabilities.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import openvino as ov
import torch

import torch.nn.functional as F

from action_clf.model import build_action_classifier, normalize_clips


class ClipClassifier(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, frames: int, stored_size: int, input_size: int) -> None:
        super().__init__()
        self.model = model
        self.frames = frames
        self.input_size = input_size
        # train_action_clips.center_view with sizes fixed up front: tracing cannot round a shape.
        self.side = int(round(stored_size * 0.85))
        self.offset = (stored_size - self.side) // 2

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        clips = normalize_clips(clips)
        clips = clips[..., self.offset : self.offset + self.side, self.offset : self.offset + self.side]
        flat = clips.permute(0, 2, 1, 3, 4).reshape(-1, 3, self.side, self.side)
        flat = F.interpolate(flat, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        clips = flat.reshape(-1, self.frames, 3, self.input_size, self.input_size).permute(0, 2, 1, 3, 4)
        return self.model(clips).softmax(-1)


def main() -> None:
    checkpoint_path, output_stem = Path(sys.argv[1]), Path(sys.argv[2])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = build_action_classifier(checkpoint["arch"], len(class_names), pretrained=False, dropout=checkpoint["dropout"])
    model.load_state_dict(checkpoint["model"])
    frames = 2 * checkpoint["half_window"] + 1
    size = checkpoint["stored_size"]
    wrapped = ClipClassifier(model, frames, size, checkpoint["input_size"]).eval()
    example = torch.randint(0, 256, (2, frames, size, size, 3)).float()
    onnx_path = output_stem.with_suffix(".onnx")
    torch.onnx.export(
        wrapped,
        example,
        onnx_path,
        input_names=["clips"],
        output_names=["probabilities"],
        dynamic_axes={"clips": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, output_stem.with_suffix(".xml"), compress_to_fp16=True)
    onnx_path.unlink()

    meta = {key: checkpoint[key] for key in ("arch", "class_names", "half_window", "stored_size", "crop_scale", "input_size")}
    # How the crop window is built ("player": sized by the player's height); older checkpoints have no key.
    if "crop" in checkpoint:
        meta["crop"] = checkpoint["crop"]
    meta["source_checkpoint"] = checkpoint_path.name
    output_stem.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    from train_action_clips import center_view

    with torch.no_grad():
        expected = model(center_view(normalize_clips(example), checkpoint["input_size"])).softmax(-1).numpy()
    actual = ov.Core().compile_model(ov_model, "CPU")(example.numpy())[0]
    print(f"saved {output_stem}.xml, max probability difference vs torch: {abs(expected - actual).max():.5f}")


if __name__ == "__main__":
    main()
