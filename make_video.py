#!/usr/bin/env python3
"""Create a video from PNG images in a directory, using ffmpeg via pipe."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path: Path) -> np.ndarray | None:
    try:
        pil_img = Image.open(img_path).convert("RGB")
        return np.array(pil_img)  # RGB, uint8
    except (UnidentifiedImageError, Exception):
        return None


def make_video(input_dir: str, output_path: str, fps: float = 24.0) -> None:
    img_dir = Path(input_dir)
    out_path = Path(output_path)

    images = sorted(img_dir.glob("*.png"), key=lambda p: p.name)
    if not images:
        print(f"No PNG files found in {img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images")

    # Find first readable image to get dimensions
    h, w = None, None
    for probe in images:
        frame = read_image(probe)
        if frame is not None:
            h, w = frame.shape[:2]
            print(f"Dimensions from {probe.name}: {w}x{h}")
            break
    if h is None:
        print("Could not read any image to determine dimensions", file=sys.stderr)
        sys.exit(1)

    # H.264 requires dimensions divisible by 2
    w = w - (w % 2)
    h = h - (h % 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    skipped = 0
    written = 0
    for img_path in images:
        frame = read_image(img_path)
        if frame is None:
            print(f"  skipping unreadable: {img_path.name}")
            skipped += 1
            continue
        if frame.shape[:2] != (h, w):
            from PIL import Image as PILImage
            pil = PILImage.fromarray(frame).resize((w, h))
            frame = np.array(pil)
        proc.stdin.write(frame.tobytes())
        written += 1
        if written % 100 == 0:
            print(f"  {written} frames written...")

    proc.stdin.close()
    proc.wait()

    print(f"Done — {written} frames written, {skipped} skipped")
    print(f"Video saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from PNG images.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/Senecio_17_11_L1_Cavicam16_210725",
        help="Directory containing PNG images",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="videos/Senecio_17_11_L1_Cavicam16_210725.mp4",
        help="Output video path",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="Frames per second (default: 24)")
    args = parser.parse_args()

    make_video(args.input_dir, args.output, args.fps)
