#!/usr/bin/env python3
"""
check_data.py — Data integrity check for all leaf sequences.

For each sequence directory reports:
  - Number of PNG frames (excluding test_image_uncropped.png)
  - Number of GT mask TIFFs in the *_analysedStack subfolder
  - Mismatch between frames-1 and masks (expected: n_frames-1 == n_masks)
  - Empty TIFF files (0 bytes)
  - Truncated TIFF files (declared strip size > actual file size)
  - TIFF files with unreadable headers

Usage (from project root):
    python src/check_data.py
    python src/check_data.py --data-root /other/path
    python src/check_data.py --sequence Senecio_19_04_L5_Cavicam05_160725
"""

import argparse
import os
import struct
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    from scipy.ndimage import label as nd_label
    _HAS_IMAGE_LIBS = True
except ImportError:
    _HAS_IMAGE_LIBS = False


# ---------------------------------------------------------------------------
# File listing (mirrors dataset.py logic exactly)
# ---------------------------------------------------------------------------

def _list_frames(seq_dir: Path):
    ignore = {"test_image_uncropped.png"}
    return sorted(
        f for f in seq_dir.glob("*.png")
        if f.name not in ignore and not f.name.endswith(".bak")
    )


def _list_masks(seq_dir: Path):
    stack_dirs = [d for d in seq_dir.iterdir() if d.is_dir() and "analysedstack" in d.name.lower()]
    if not stack_dirs:
        return []
    return sorted(stack_dirs[0].glob("*.tif"))


# ---------------------------------------------------------------------------
# TIFF integrity check
# ---------------------------------------------------------------------------

def check_tiff(path: Path) -> dict:
    """
    Returns a dict with keys:
      ok        : bool — True if file looks valid
      empty     : bool — file is 0 bytes
      bad_header: bool — file too short to contain a valid TIFF header
      truncated : bool — declared strip byte count > actual bytes available
      declared  : int  — declared strip byte count
      available : int  — actual bytes available from strip offset to EOF
      error     : str  — exception message if parsing failed unexpectedly
    """
    result = dict(ok=False, empty=False, bad_header=False,
                  truncated=False, declared=0, available=0, error="")

    size = path.stat().st_size
    if size == 0:
        result["empty"] = True
        return result

    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        result["error"] = str(e)
        return result

    if len(data) < 8:
        result["bad_header"] = True
        return result

    try:
        endian = "<" if data[:2] == b"II" else ">"
        ifd_off = struct.unpack_from(endian + "I", data, 4)[0]
        n = struct.unpack_from(endian + "H", data, ifd_off)[0]
        tags = {}
        for i in range(n):
            off = ifd_off + 2 + i * 12
            tag, dtype, count = struct.unpack_from(endian + "HHI", data, off)
            vo = off + 8
            if count == 1:
                val = struct.unpack_from(endian + ("H" if dtype == 3 else "I"), data, vo)[0]
            else:
                ptr = struct.unpack_from(endian + "I", data, vo)[0]
                fmt = "H" if dtype == 3 else "I"
                val = list(struct.unpack_from(endian + f"{count}{fmt}", data, ptr))
            tags[tag] = (dtype, count, val)

        s_off = tags[273][2]; s_off = s_off[0] if isinstance(s_off, list) else s_off
        s_len = tags[279][2]; s_len = s_len[0] if isinstance(s_len, list) else s_len

        available = max(0, len(data) - s_off)
        result["declared"]  = s_len
        result["available"] = available

        if s_len > available:
            result["truncated"] = True
        else:
            result["ok"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Mask content statistics
# ---------------------------------------------------------------------------

def _mask_stats(mask_path: Path) -> dict:
    """
    Returns dict with:
      n_events      : int  — number of connected components (events) in the mask
      n_active_px   : int  — number of non-zero pixels
      total_px      : int  — total pixels in the mask
      readable      : bool — False if the image could not be opened
    """
    try:
        arr = np.array(Image.open(mask_path))
        binary = arr > 0
        total_px = arr.size
        n_active_px = int(binary.sum())
        _, n_events = nd_label(binary)
        return dict(readable=True, n_events=n_events, n_active_px=n_active_px, total_px=total_px)
    except Exception:
        return dict(readable=False, n_events=0, n_active_px=0, total_px=0)


# ---------------------------------------------------------------------------
# Per-sequence check
# ---------------------------------------------------------------------------

def check_sequence(seq_dir: Path) -> dict:
    frames = _list_frames(seq_dir)
    masks  = _list_masks(seq_dir)

    n_frames = len(frames)
    n_masks  = len(masks)
    expected = n_frames - 1  # one mask per consecutive pair

    issues = []
    empty_tiffs     = []
    truncated_tiffs = []
    bad_tiffs       = []

    # content stats (filled only when image libs are available)
    masks_with_events   = 0
    masks_without_events = 0
    total_events        = 0
    total_active_px     = 0
    total_px_all        = 0

    if n_masks == 0:
        issues.append("NO _analysedStack subfolder or no .tif files found")
    elif n_masks != expected:
        issues.append(f"frame pairs={expected}, masks={n_masks} (diff={n_masks - expected:+d})")

    for mask_path in masks:
        r = check_tiff(mask_path)
        if r["empty"]:
            empty_tiffs.append(mask_path.name)
        elif r["bad_header"]:
            bad_tiffs.append(mask_path.name)
        elif r["error"]:
            bad_tiffs.append(f"{mask_path.name} ({r['error']})")
        elif r["truncated"]:
            truncated_tiffs.append(
                f"{mask_path.name} (declared={r['declared']}, available={r['available']})"
            )

        if _HAS_IMAGE_LIBS and r.get("ok", False):
            ms = _mask_stats(mask_path)
            if ms["readable"]:
                total_events   += ms["n_events"]
                total_active_px += ms["n_active_px"]
                total_px_all   += ms["total_px"]
                if ms["n_events"] > 0:
                    masks_with_events += 1
                else:
                    masks_without_events += 1

    return {
        "n_frames":            n_frames,
        "n_masks":             n_masks,
        "expected_masks":      expected,
        "issues":              issues,
        "empty_tiffs":         empty_tiffs,
        "truncated_tiffs":     truncated_tiffs,
        "bad_tiffs":           bad_tiffs,
        "masks_with_events":   masks_with_events,
        "masks_without_events": masks_without_events,
        "total_events":        total_events,
        "total_active_px":     total_active_px,
        "total_px_all":        total_px_all,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(seq_name: str, r: dict, verbose: bool = False) -> bool:
    """Print report for one sequence. Returns True if any problem was found."""
    has_issue = (
        r["issues"] or r["empty_tiffs"] or r["bad_tiffs"] or
        (r["truncated_tiffs"] and verbose)
    )

    n_trunc = len(r["truncated_tiffs"])

    if not has_issue and not verbose:
        # Compact OK line — only show truncated count as a note
        trunc_note = f"  [{n_trunc} truncated TIFFs — padded by loader]" if n_trunc else ""
        print(f"  OK  {seq_name}  "
              f"(frames={r['n_frames']}, masks={r['n_masks']}){trunc_note}")
        return False

    # Detailed report
    status = "WARN" if not r["issues"] and not r["empty_tiffs"] and not r["bad_tiffs"] else "FAIL"
    print(f"\n  {status}  {seq_name}")
    print(f"        frames={r['n_frames']}  masks={r['n_masks']}  "
          f"expected={r['expected_masks']}")

    for msg in r["issues"]:
        print(f"        ! {msg}")
    if r["empty_tiffs"]:
        print(f"        ! {len(r['empty_tiffs'])} empty TIFF(s) (0 bytes):")
        for f in r["empty_tiffs"]:
            print(f"            {f}")
    if r["bad_tiffs"]:
        print(f"        ! {len(r['bad_tiffs'])} unreadable TIFF(s):")
        for f in r["bad_tiffs"]:
            print(f"            {f}")
    if r["truncated_tiffs"] and verbose:
        print(f"        ~ {n_trunc} truncated TIFF(s) (handled by zero-padding):")
        for f in r["truncated_tiffs"]:
            print(f"            {f}")
    elif n_trunc:
        print(f"        ~ {n_trunc} truncated TIFF(s) — padded by loader (use --verbose to list)")

    return status == "FAIL"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check data integrity for leaf sequences.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sequence", "-s", default=None,
                        help="Check a single sequence by name")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Also list truncated TIFFs (expected — handled by loader)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent
    data_root  = Path(args.data_root) if args.data_root else repo_root / "data"

    if not data_root.is_dir():
        raise SystemExit(f"[ERROR] data root not found: {data_root}")

    # Collect sequences
    if args.sequence:
        seq_dirs = [data_root / args.sequence]
        if not seq_dirs[0].is_dir():
            raise SystemExit(f"[ERROR] Sequence not found: {seq_dirs[0]}")
    else:
        seq_dirs = sorted(
            d for d in data_root.iterdir()
            if d.is_dir() and list(d.glob("*.png"))
        )

    print(f"\nData root: {data_root}")
    print(f"Checking {len(seq_dirs)} sequence(s) …\n")

    total_fails   = 0
    total_empty   = 0
    total_bad     = 0
    total_trunc   = 0
    total_frames  = 0
    total_masks   = 0
    g_with_ev     = 0
    g_without_ev  = 0
    g_events      = 0
    g_active_px   = 0
    g_total_px    = 0

    for seq_dir in seq_dirs:
        r = check_sequence(seq_dir)
        failed = print_report(seq_dir.name, r, verbose=args.verbose)
        if failed:
            total_fails += 1
        total_empty  += len(r["empty_tiffs"])
        total_bad    += len(r["bad_tiffs"])
        total_trunc  += len(r["truncated_tiffs"])
        total_frames += r["n_frames"]
        total_masks  += r["n_masks"]
        g_with_ev    += r["masks_with_events"]
        g_without_ev += r["masks_without_events"]
        g_events     += r["total_events"]
        g_active_px  += r["total_active_px"]
        g_total_px   += r["total_px_all"]

    print(f"\n{'─'*60}")
    print(f"Summary: {len(seq_dirs)} sequences checked")
    print(f"  Sequences with errors : {total_fails}")
    print(f"  Empty TIFFs (0 bytes) : {total_empty}")
    print(f"  Unreadable TIFFs      : {total_bad}")
    print(f"  Truncated TIFFs       : {total_trunc}  (handled by zero-padding)")

    print(f"\n{'─'*60}")
    print(f"Dataset statistics")
    print(f"  Total images (frames) : {total_frames}")
    print(f"  Total masks           : {total_masks}")

    if _HAS_IMAGE_LIBS and g_total_px > 0:
        readable_masks = g_with_ev + g_without_ev
        pct_with    = 100.0 * g_with_ev    / readable_masks if readable_masks else 0.0
        pct_without = 100.0 * g_without_ev / readable_masks if readable_masks else 0.0
        avg_ev_per_mask = g_events / readable_masks if readable_masks else 0.0
        avg_active_px   = g_active_px / readable_masks if readable_masks else 0.0
        pct_active_px   = 100.0 * g_active_px / g_total_px if g_total_px else 0.0

        print(f"  Masks with ≥1 event   : {g_with_ev}  ({pct_with:.1f}%)")
        print(f"  Masks with 0 events   : {g_without_ev}  ({pct_without:.1f}%)")
        print(f"  Avg events / mask     : {avg_ev_per_mask:.2f}")
        print(f"  Avg active px / mask  : {avg_active_px:.1f}  ({pct_active_px:.4f}% of mask area)")
    elif not _HAS_IMAGE_LIBS:
        print("  [content stats unavailable — install numpy, Pillow, scipy]")

    if total_fails > 0 or total_empty > 0 or total_bad > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
