#!/usr/bin/env python3
"""
interpolate_video.py
====================
Usage:
    cd /workspace && export PYTHONPATH=$PYTHONPATH:/workspace/IFRNet && python interpolate_video.py --input ./ShortsAI/output.mp4 --target_fps 24 --model ./IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth
    python interpolate_video.py --input output.mp4 --target_fps 24 --model IFRNet_Vimeo90K.pth

- Detects the source video's FPS automatically.
- Computes how many intermediate frames are needed to reach the target FPS.
- Uses IFRNet for frame interpolation (CUDA required).
"""

import argparse
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm

# --- IFRNet import (make sure you're inside or have IFRNet in PYTHONPATH)
from models.IFRNet import Model as IFRNet


def load_model(model_path: str):
    print(f"[INFO] Loading IFRNet weights from {model_path} ...")
    model = IFRNet().cuda().eval()
    checkpoint = torch.load(model_path, map_location="cuda")
    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print("[INFO] Model loaded.")
    return model


@torch.inference_mode()
def interpolate_frame(model, img0, img1, t):
    I0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
    I1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
    time = torch.tensor([t], device="cuda").view(1, 1, 1, 1).float()
    It = model.inference(I0, I1, time)[0]
    out = (It.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--target_fps", type=float, required=True, help="Desired FPS (e.g., 24)")
    parser.add_argument("--model", default="pretrained/IFRNet_Vimeo90K.pth", help="Path to IFRNet checkpoint")
    parser.add_argument("--output", default="out_interpolated.mp4", help="Output video path")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Read video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Input FPS: {fps:.2f}, target FPS: {args.target_fps:.2f}")

    factor = args.target_fps / fps
    if factor <= 1:
        print("[WARN] Target FPS ≤ source FPS, just copying frames.")
        factor = 1

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames.append(frame)
    cap.release()

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.target_fps,
        (width, height)
    )

    print(f"[INFO] Interpolating with factor ≈ {factor:.2f}x")
    interp_count = int(round(factor)) - 1

    for i in tqdm(range(len(frames) - 1)):
        f0, f1 = frames[i], frames[i + 1]
        out.write(f0)
        for k in range(1, interp_count + 1):
            t = k / (interp_count + 1)
            mid = interpolate_frame(model, f0, f1, t)
            out.write(mid)
    out.write(frames[-1])
    out.release()

    print(f"[INFO] Done → {args.output}")


if __name__ == "__main__":
    main()
