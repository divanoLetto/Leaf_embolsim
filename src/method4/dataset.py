"""
dataset.py — Method 4: Dataset with physics-calibrated augmentation.

Augmentation rationale (Cavicam / RPi Camera v2, 5-minute intervals)
---------------------------------------------------------------------
The camera uses its own fixed LED illumination with fixed ISO and shutter speed,
so brightness is stable within a session. The main real-world variation sources are:

  1. Ambient light leaking through the greenhouse/field setup → small global
     brightness offset, SHARED between the two frames (both frames see the same
     ambient light at their respective times, but 5 minutes apart the ambient can
     differ slightly).

  2. Shot noise from the IMX219 sensor → uncorrelated between frames, so applied
     INDEPENDENTLY to frame_t and frame_{t+1}.

  3. Local tissue deformation from turgour changes over 5 minutes → elastic
     deformation applied JOINTLY to both frames and the mask so they stay aligned.

  4. Random spatial flips and 90° rotations → JOINT, increase inter-sequence
     geometric diversity (different mount angles, different leaf orientations).

Functions that changed vs Method 2
------------------------------------
  _augment() — completely rewritten with the calibrated augmentation strategy.
  Everything else is identical to Method 2.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method4 import config


# ---------------------------------------------------------------------------
# Low-level I/O helpers  (unchanged from Method 2)
# ---------------------------------------------------------------------------

def _list_frames(seq_dir: str) -> List[str]:
    p = Path(seq_dir)
    ignore = {"test_image_uncropped.png"}
    return sorted([
        str(f) for f in p.glob("*.png")
        if f.name not in ignore and not f.name.endswith(".bak")
    ])


def _list_masks(seq_dir: str) -> List[str]:
    # Masks always live in an "analysedStack" folder. The "substackforanalysis"
    # folder, when present, contains the PhD's cropped input images (grayscale,
    # not binary) and must NOT be used as masks.
    p = Path(seq_dir)
    stack_dirs = [d for d in p.iterdir() if d.is_dir() and "analysedstack" in d.name.lower()]
    if not stack_dirs:
        return []
    return sorted(str(f) for f in stack_dirs[0].glob("*.tif"))


def _npy_cache_dir(seq_dir: str) -> Path:
    d = Path(seq_dir) / ".npy_cache"
    d.mkdir(exist_ok=True)
    return d


def _safe_npy_save(arr: np.ndarray, npy_path: Path) -> None:
    try:
        np.save(str(npy_path), arr)
        if npy_path.exists() and npy_path.stat().st_size == 0:
            npy_path.unlink()
    except Exception:
        try:
            npy_path.unlink()
        except OSError:
            pass


def _load_rgb(path: str, h: int, w: int, cache_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    if cache_dir is not None:
        npy_path = cache_dir / (Path(path).stem + "_rgb.npy")
        if npy_path.exists() and npy_path.stat().st_size > 0:
            return np.load(str(npy_path))

    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
    except Exception as exc:
        print(f"  [WARN] Cannot load frame {Path(path).name}: {exc}")
        return None

    if cache_dir is not None:
        _safe_npy_save(arr, npy_path)
    return arr


def _load_mask(path: str, h: int, w: int, cache_dir: Optional[Path] = None) -> np.ndarray:
    if cache_dir is not None:
        npy_path = cache_dir / (Path(path).stem + "_mask.npy")
        if npy_path.exists() and npy_path.stat().st_size > 0:
            return np.load(str(npy_path))

    arr = _decode_tiff_mask(path, h, w)

    if cache_dir is not None:
        _safe_npy_save(arr, npy_path)
    return arr


def _decode_tiff_mask(path: str, h: int, w: int) -> np.ndarray:
    import struct, zlib

    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        print(f"  [WARN] Cannot open mask {Path(path).name}: {exc}")
        return np.zeros((h, w), dtype=np.uint8)

    if len(data) < 8:
        print(f"  [WARN] Mask {Path(path).name} is empty or too short ({len(data)} bytes) — treating as all-background.")
        return np.zeros((h, w), dtype=np.uint8)

    endian = "<" if data[:2] == b"II" else ">"

    ifd_off = struct.unpack_from(endian + "I", data, 4)[0]
    n       = struct.unpack_from(endian + "H", data, ifd_off)[0]
    tags    = {}
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

    src_h = tags[257][2]
    src_w = tags[256][2]
    s_off = tags[273][2]; s_off = s_off[0] if isinstance(s_off, list) else s_off
    s_len = tags[279][2]; s_len = s_len[0] if isinstance(s_len, list) else s_len
    comp  = tags.get(259, (None, None, 1))[2]

    avail = max(0, len(data) - s_off)
    raw   = data[s_off: s_off + min(s_len, avail)]
    if comp in (8, 32946):
        try: raw = zlib.decompress(raw)
        except: pass

    expected = src_h * src_w
    arr = np.frombuffer(raw, dtype=np.uint8)
    if len(arr) < expected:
        pad = np.zeros(expected, dtype=np.uint8)
        pad[:len(arr)] = arr
        arr = pad
    else:
        arr = arr[:expected]

    arr    = arr.reshape(src_h, src_w)
    binary = (arr >= 128).astype(np.uint8)

    img = Image.fromarray(binary * 255).resize((w, h), Image.NEAREST)
    return (np.array(img) > 128).astype(np.uint8)


def _check_dim_mismatch(image_path: str, mask_path: str) -> Optional[Tuple[int, int, int, int]]:
    """Return (iw, ih, mw, mh) if image and mask native dims differ, else None."""
    import struct
    try:
        iw, ih = Image.open(image_path).size
        with open(mask_path, "rb") as fh:
            data = fh.read(4096)
        endian = "<" if data[:2] == b"II" else ">"
        ifd_off = struct.unpack_from(endian + "I", data, 4)[0]
        n = struct.unpack_from(endian + "H", data, ifd_off)[0]
        mw = mh = None
        for i in range(n):
            off = ifd_off + 2 + i * 12
            tag, _, _ = struct.unpack_from(endian + "HHI", data, off)
            if tag == 256:
                mw = struct.unpack_from(endian + "I", data, off + 8)[0]
            elif tag == 257:
                mh = struct.unpack_from(endian + "I", data, off + 8)[0]
        if mw is None or mh is None:
            return None
        if (iw, ih) != (mw, mh):
            return iw, ih, mw, mh
        return None
    except Exception:
        return None


def find_sequences(data_root: str) -> List[Tuple[str, str]]:
    return sorted(
        [(e.name, str(e)) for e in Path(data_root).iterdir()
         if e.is_dir() and list(e.glob("*.png"))],
        key=lambda x: x[0],
    )


def load_split(split_file: str) -> List[str]:
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Run  python src/make_splits.py  from the project root to generate it."
        )
    seqs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                seqs.append(line)
    return seqs


# ---------------------------------------------------------------------------
# Augmentation  (new in Method 4)
# ---------------------------------------------------------------------------

def _elastic_deformation(
    a: np.ndarray,
    b: np.ndarray,
    m: np.ndarray,
    alpha: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the same smooth random elastic deformation to both frames and the mask.

    The displacement field is generated by smoothing random noise with a
    Gaussian kernel (scipy), then scaling by alpha.  Both frames and the mask
    receive the SAME field so they remain spatially aligned.

    Uses scipy.ndimage which is always available in the scientific Python stack.
    If scipy is not installed, this augmentation is silently skipped.
    """
    try:
        from scipy.ndimage import gaussian_filter, map_coordinates
    except ImportError:
        return a, b, m

    h, w = a.shape[:2]
    rng  = np.random.default_rng()

    # Random displacement fields, smoothed to produce physically plausible deformation
    dx = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma=sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma=sigma) * alpha

    # Coordinate grids
    y, x   = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords_y = np.clip(y + dy, 0, h - 1)
    coords_x = np.clip(x + dx, 0, w - 1)
    coords   = [coords_y.ravel(), coords_x.ravel()]

    def _warp_rgb(img):
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[..., c] = map_coordinates(
                img[..., c], coords, order=1, mode="reflect"
            ).reshape(h, w)
        return out

    def _warp_mask(msk):
        warped = map_coordinates(msk.astype(np.float32), coords, order=0, mode="reflect")
        return (warped.reshape(h, w) > 0.5).astype(np.uint8)

    return _warp_rgb(a), _warp_rgb(b), _warp_mask(m)


def _augment(
    a: np.ndarray,
    b: np.ndarray,
    m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply calibrated augmentation to a (frame_t, frame_{t+1}, mask) triplet.

    Joint transforms (applied identically to a, b, m — spatial alignment preserved):
      - Horizontal flip          (p=0.5)
      - Vertical flip            (p=0.5)
      - 90° rotation × k        (k ∈ {0,1,2,3}, uniform)
      - Fine rotation            U(AUG_FINE_ROT_RANGE) degrees — simulates mount
                                 placement variability between plants/sessions
      - Elastic deformation      (p=AUG_ELASTIC_PROB)

    Shared photometric transforms (same value applied to both a and b —
    simulates inter-camera and inter-session variability, not intra-sequence drift):
      - Gamma correction         U(AUG_GAMMA_RANGE) — different IMX219 units have
                                 slightly different nonlinear responses
      - Channel shift            U(AUG_CHANNEL_SHIFT) per RGB channel — small WB
                                 differences between sessions/mounts

    Independent transforms (applied separately to a and b):
      - Brightness scale         U(AUG_BRIGHTNESS_RANGE) per frame
      - Gaussian noise           sigma ~ U(AUG_NOISE_SIGMA_RANGE) per frame, uncorrelated

    NOT applied:
      - Motion blur              (fixed focus, stable LED — no physical justification)
    """
    # ── Joint spatial transforms ────────────────────────────────────────────
    if random.random() < 0.5:
        a, b, m = np.fliplr(a), np.fliplr(b), np.fliplr(m)
    if random.random() < 0.5:
        a, b, m = np.flipud(a), np.flipud(b), np.flipud(m)
    k = random.randint(0, 3)
    if k > 0:
        a = np.rot90(a, k)
        b = np.rot90(b, k)
        m = np.rot90(m, k)

    # Fine rotation: simulates imperfect mount alignment between plants/sessions.
    # Applied after rot90 so the two work together. reflect padding avoids black
    # corners; with angles ≤10° the reflected area is only a few pixels wide.
    angle = random.uniform(*config.AUG_FINE_ROT_RANGE)
    if abs(angle) > 0.5:
        try:
            from scipy.ndimage import rotate as _nd_rotate
            a = _nd_rotate(a, angle, axes=(0, 1), reshape=False, mode="reflect", order=1)
            b = _nd_rotate(b, angle, axes=(0, 1), reshape=False, mode="reflect", order=1)
            m = (_nd_rotate(m.astype(np.float32), angle, axes=(0, 1),
                            reshape=False, mode="reflect", order=0) > 0.5).astype(np.uint8)
        except ImportError:
            pass

    if random.random() < config.AUG_ELASTIC_PROB:
        a, b, m = _elastic_deformation(
            a, b, m,
            alpha=config.AUG_ELASTIC_ALPHA,
            sigma=config.AUG_ELASTIC_SIGMA,
        )

    # ── Shared photometric transforms (inter-camera/inter-session variability) ─
    # Gamma: same nonlinear response shift for both frames — simulates different
    # IMX219 units having slightly different gamma curves.
    gamma = random.uniform(*config.AUG_GAMMA_RANGE)
    a = np.clip(np.power(a.astype(np.float32) / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
    b = np.clip(np.power(b.astype(np.float32) / 255.0, gamma) * 255, 0, 255).astype(np.uint8)

    # Channel shift: same per-channel offset for both frames — simulates small
    # white-balance differences between sessions or mounts.
    shift = np.random.uniform(*config.AUG_CHANNEL_SHIFT, size=(3,)).astype(np.float32)
    a = np.clip(a.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) + shift, 0, 255).astype(np.uint8)

    # ── Independent per-frame transforms ───────────────────────────────────
    lo, hi = config.AUG_BRIGHTNESS_RANGE
    scale_a = random.uniform(lo, hi)
    scale_b = random.uniform(lo, hi)
    a = np.clip(a.astype(np.float32) * scale_a, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * scale_b, 0, 255).astype(np.uint8)

    sigma_lo, sigma_hi = config.AUG_NOISE_SIGMA_RANGE
    sigma_a = random.uniform(sigma_lo, sigma_hi)
    sigma_b = random.uniform(sigma_lo, sigma_hi)
    noise_a  = np.random.normal(0, sigma_a, a.shape)
    noise_b  = np.random.normal(0, sigma_b, b.shape)
    a = np.clip(a.astype(np.float32) + noise_a, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) + noise_b, 0, 255).astype(np.uint8)

    return (
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        np.ascontiguousarray(m),
    )


# ---------------------------------------------------------------------------
# Core dataset class  (identical to Method 2 except _augment call)
# ---------------------------------------------------------------------------

class LeafPairDataset:
    def __init__(self, seq_names: List[str], data_root: str, augment: bool = False):
        self.data_root = data_root
        self.augment   = augment
        self.h         = config.IMG_HEIGHT
        self.w         = config.IMG_WIDTH
        self.patch     = config.PATCH_SIZE

        self.triplets:     List[Tuple[str, str, str]] = []
        self.has_embolism: List[bool]                 = []

        for seq_name in seq_names:
            seq_path  = os.path.join(data_root, seq_name)
            cache_dir = _npy_cache_dir(seq_path)
            frames    = _list_frames(seq_path)
            masks     = _list_masks(seq_path)

            n_frames = len(frames)
            n_masks  = len(masks)
            n_pairs  = min(n_frames - 1, n_masks) if n_masks > 0 else 0

            if n_masks == 0:
                print(f"  [WARN] Sequence '{seq_name}': NO GT masks found — skipped.")
                continue

            # Sanity check: raw image and raw mask must have matching native
            # dimensions, otherwise independent resize() in _load_rgb / _load_mask
            # maps them to incompatible coordinates and the mask ends up
            # overlaid on the wrong region of the image.
            if frames and masks:
                mismatch = _check_dim_mismatch(frames[0], masks[0])
                if mismatch is not None:
                    iw, ih, mw, mh = mismatch
                    print(
                        f"  [WARN] Sequence '{seq_name}': image native size "
                        f"{iw}x{ih} != mask native size {mw}x{mh}. "
                        f"Independent resize will misalign mask vs image — "
                        f"training on this sequence is UNSAFE."
                    )

            if n_frames - 1 != n_masks:
                print(
                    f"  [WARN] Sequence '{seq_name}': "
                    f"{n_frames} frames → {n_frames-1} pairs, but {n_masks} GT masks. "
                    f"Using first {n_pairs} pairs."
                )
            else:
                print(f"  Sequence '{seq_name}': {n_frames} frames, "
                      f"{n_masks} masks → {n_pairs} pairs.")

            he_cache_path = cache_dir / "has_embolism.json"
            if he_cache_path.exists():
                with open(he_cache_path) as fh:
                    cached = json.load(fh)
                he_flags = cached.get("flags", [])
                if len(he_flags) >= n_pairs:
                    he_flags = he_flags[:n_pairs]
                    print(f"    → has_embolism cache hit ({n_pairs} pairs).")
                else:
                    he_flags = None
                    print(f"    → has_embolism cache stale, rebuilding …")
            else:
                he_flags = None

            if he_flags is None:
                print(f"    → Scanning {n_pairs} masks for embolism events …")
                he_flags = []
                for i in range(n_pairs):
                    mask_arr = _load_mask(masks[i], self.h, self.w, cache_dir)
                    he_flags.append(bool(mask_arr.any()))
                with open(he_cache_path, "w") as fh:
                    json.dump({"flags": he_flags, "n_pairs": n_pairs}, fh)
                print(f"    → Saved has_embolism cache.")

            n_embolism = sum(he_flags)
            print(f"    → {n_pairs} pairs, "
                  f"{n_embolism} with embolism events ({100*n_embolism/max(n_pairs,1):.1f}%).")

            for i in range(n_pairs):
                self.triplets.append((frames[i], frames[i + 1], masks[i]))
                self.has_embolism.append(he_flags[i])

        self._sample_index: List[int] = []
        for i, has_ev in enumerate(self.has_embolism):
            w = config.OVERSAMPLE_FACTOR if has_ev else 1
            self._sample_index.extend([i] * w)

        print(f"  Total pairs: {len(self.triplets)}, "
              f"weighted pool size: {len(self._sample_index)}")

        self._prewarm_cache()

    def _prewarm_cache(self) -> None:
        all_frames: set = set()
        all_masks:  set = set()
        for f_t, f_t1, m_t in self.triplets:
            all_frames.add(f_t)
            all_frames.add(f_t1)
            all_masks.add(m_t)

        def _needs_cache(path: str, suffix: str) -> bool:
            p = _npy_cache_dir(str(Path(path).parent)) / (Path(path).stem + suffix)
            return not p.exists() or p.stat().st_size == 0

        missing_frames = [p for p in all_frames if _needs_cache(p, "_rgb.npy")]
        missing_masks  = [p for p in all_masks  if _needs_cache(p, "_mask.npy")]
        total = len(missing_frames) + len(missing_masks)

        if total == 0:
            print("  .npy cache: fully warm, nothing to do.")
            return

        print(f"  .npy cache: warming {len(missing_frames)} frames + "
              f"{len(missing_masks)} masks in main process …")

        for i, path in enumerate(sorted(missing_frames), 1):
            cache_dir = _npy_cache_dir(str(Path(path).parent))
            _load_rgb(path, self.h, self.w, cache_dir)
            if i % 200 == 0:
                print(f"    frames {i}/{len(missing_frames)} …")

        for i, path in enumerate(sorted(missing_masks), 1):
            cache_dir = _npy_cache_dir(str(Path(path).parent))
            _load_mask(path, self.h, self.w, cache_dir)
            if i % 200 == 0:
                print(f"    masks {i}/{len(missing_masks)} …")

        print("  .npy cache: warm.")

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: int):
        import torch

        triplet_idx     = self._sample_index[idx]
        f_t, f_t1, m_t = self.triplets[triplet_idx]
        has_ev          = self.has_embolism[triplet_idx]

        rgb_t  = _load_rgb(f_t,  self.h, self.w, _npy_cache_dir(str(Path(f_t).parent)))
        rgb_t1 = _load_rgb(f_t1, self.h, self.w, _npy_cache_dir(str(Path(f_t1).parent)))
        mask   = _load_mask(m_t, self.h, self.w, _npy_cache_dir(str(Path(m_t).parent)))

        if rgb_t  is None: rgb_t  = np.zeros((self.h, self.w, 3), np.uint8)
        if rgb_t1 is None: rgb_t1 = np.zeros((self.h, self.w, 3), np.uint8)

        # Patch extraction
        patch = self.patch
        if has_ev and mask.any():
            ys, xs = np.where(mask == 1)
            choice = random.randrange(len(ys))
            cy, cx = int(ys[choice]), int(xs[choice])
            jitter = config.PATCH_JITTER
            cy = cy + random.randint(-jitter, jitter)
            cx = cx + random.randint(-jitter, jitter)
        else:
            cy = random.randint(0, self.h - 1)
            cx = random.randint(0, self.w - 1)

        y0 = max(0, min(cy - patch // 2, self.h - patch))
        x0 = max(0, min(cx - patch // 2, self.w - patch))
        y1, x1 = y0 + patch, x0 + patch

        p_t  = rgb_t [y0:y1, x0:x1, :]
        p_t1 = rgb_t1[y0:y1, x0:x1, :]
        p_m  = mask  [y0:y1, x0:x1]

        if self.augment:
            p_t, p_t1, p_m = _augment(p_t, p_t1, p_m)

        t_t  = torch.from_numpy(p_t.transpose(2, 0, 1).astype(np.float32) / 255.0)
        t_t1 = torch.from_numpy(p_t1.transpose(2, 0, 1).astype(np.float32) / 255.0)
        t_m  = torch.from_numpy(p_m.astype(np.float32)).unsqueeze(0)

        return {"input": torch.cat([t_t, t_t1], dim=0), "target": t_m}

    def iter_pairs(self, seq_name: str):
        seq_path = os.path.join(self.data_root, seq_name)
        frames   = _list_frames(seq_path)
        masks    = _list_masks(seq_path)
        n_pairs  = min(len(frames) - 1, len(masks))
        for i in range(n_pairs):
            rgb_t  = _load_rgb(frames[i],   self.h, self.w, _npy_cache_dir(str(Path(frames[i]).parent)))
            rgb_t1 = _load_rgb(frames[i+1], self.h, self.w, _npy_cache_dir(str(Path(frames[i+1]).parent)))
            mask   = _load_mask(masks[i],   self.h, self.w, _npy_cache_dir(str(Path(masks[i]).parent)))
            if rgb_t  is None: rgb_t  = np.zeros((self.h, self.w, 3), np.uint8)
            if rgb_t1 is None: rgb_t1 = np.zeros((self.h, self.w, 3), np.uint8)
            yield rgb_t, rgb_t1, mask, i


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loader(seq_names, data_root, augment, batch_size, num_workers=8, shuffle=True):
    import torch
    from torch.utils.data import DataLoader

    ds = LeafPairDataset(seq_names, data_root, augment=augment)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return loader, ds
