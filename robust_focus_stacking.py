"""
Robust Multi-Focus Image Fusion for Microscopy
================================================
A research-grade focus stacking pipeline that produces an all-in-focus
composite from a focus stack captured at different focal depths and
magnifications.  Two fusion strategies are implemented and compared:

  Strategy A  – Laplacian Pyramid Fusion   (multi-band, seamless)
  Strategy B  – Weighted-Average Fusion    (traditional, for comparison)

Ghost-artifact suppression, spatial consistency enforcement, and
post-processing (unsharp-mask sharpening + CLAHE contrast) are applied
to both strategies so differences are purely due to the blending core.

Author : Focus Stacking Research Pipeline
Date   : April 2026
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import os
import sys
import time
import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Global Configuration
# ──────────────────────────────────────────────
# Alignment
SIFT_N_FEATURES = 8000              # More features → more robust matching
RANSAC_REPROJ_THRESH = 5.0          # pixels – tolerance for inliers
MIN_MATCH_COUNT = 30                # reject alignment if fewer matches

# Focus measure
FOCUS_KERNEL_SIZE = 5               # Sobel / Laplacian kernel size
FOCUS_GAUSS_SIGMA = 7.0             # spatial smoothing of focus score map
FOCUS_GAUSS_KSIZE = 31              # kernel for the above (must be odd)

# Decision map & ghost suppression
DECISION_MORPH_KSIZE = 15           # morphological open/close kernel
DECISION_SMOOTH_SIGMA = 9.0         # final Gaussian smooth on decision map
DECISION_SMOOTH_KSIZE = 41          # kernel for the above

# Laplacian pyramid
PYRAMID_LEVELS = 6                  # number of pyramid levels

# Post-processing
SHARPEN_SIGMA = 1.0                 # unsharp-mask Gaussian sigma
SHARPEN_AMOUNT = 0.5                # strength of sharpening (0 = off)
CLAHE_CLIP_LIMIT = 2.0              # CLAHE clip limit
CLAHE_TILE_SIZE = 8                 # CLAHE grid size

# I/O
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# ==============================================================
#  1.  PREPROCESSING
# ==============================================================
def load_images(folder: str) -> list[tuple[str, np.ndarray]]:
    """Load all images from *folder*, return list of (filename, BGR image)."""
    paths = sorted(
        p
        for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            images.append((p.name, img))
        else:
            print(f"  [WARN] Could not read: {p.name}")
    return images


def to_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 → grayscale float32 in [0, 1]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Per-channel histogram stretching to [0, 255]."""
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, (0.5, 99.5))
        if hi - lo < 1.0:
            out[:, :, c] = img[:, :, c]
        else:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


# ==============================================================
#  2.  IMAGE REGISTRATION  (SIFT + RANSAC Homography)
# ==============================================================
def compute_homography(ref_gray: np.ndarray, mov_gray: np.ndarray):
    """
    Compute the 3×3 homography that warps *mov* onto *ref*
    using SIFT features + FLANN + RANSAC.

    Returns (H, n_inliers) or (None, 0) on failure.
    """
    sift = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)
    kp1, des1 = sift.detectAndCompute(
        (ref_gray * 255).astype(np.uint8) if ref_gray.dtype == np.float32 else ref_gray, None
    )
    kp2, des2 = sift.detectAndCompute(
        (mov_gray * 255).astype(np.uint8) if mov_gray.dtype == np.float32 else mov_gray, None
    )

    if des1 is None or des2 is None or len(kp1) < MIN_MATCH_COUNT or len(kp2) < MIN_MATCH_COUNT:
        return None, 0

    # FLANN matcher (KD-tree for SIFT float descriptors)
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) < MIN_MATCH_COUNT:
        return None, 0

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_mov = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_mov, pts_ref, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    n_inliers = int(mask.sum()) if mask is not None else 0

    return H, n_inliers


def warp_image(image: np.ndarray, H: np.ndarray, ref_shape: tuple) -> np.ndarray:
    """Warp *image* using homography *H* to match *ref_shape* (h, w)."""
    h, w = ref_shape[:2]
    return cv2.warpPerspective(
        image, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def align_stack(images: list[np.ndarray], ref_idx: int) -> list[np.ndarray]:
    """
    Align every image in the stack to images[ref_idx].
    Uses SIFT + Homography for robustness to focus breathing (scale changes).
    Falls back to ECC translation if SIFT fails.
    """
    ref = images[ref_idx]
    ref_gray = to_gray_float(ref)
    aligned = []

    for i, img in enumerate(images):
        if i == ref_idx:
            aligned.append(img.copy())
            continue

        mov_gray = to_gray_float(img)

        # Primary: SIFT homography
        H, n_inliers = compute_homography(ref_gray, mov_gray)
        if H is not None and n_inliers >= MIN_MATCH_COUNT:
            warped = warp_image(img, H, ref.shape)
            aligned.append(warped)
            print(f"    Image {i}: SIFT aligned ({n_inliers} inliers)")
            continue

        # Fallback: ECC translation
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            2000,
            1e-7,
        )
        try:
            ref_g8 = (ref_gray * 255).astype(np.uint8)
            mov_g8 = (mov_gray * 255).astype(np.uint8)
            _, warp_matrix = cv2.findTransformECC(
                ref_g8.astype(np.float32) / 255.0,
                mov_g8.astype(np.float32) / 255.0,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria,
            )
            h, w = ref.shape[:2]
            warped = cv2.warpAffine(
                img, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            aligned.append(warped)
            print(f"    Image {i}: ECC aligned (fallback)")
        except cv2.error:
            # Last resort: use unaligned
            aligned.append(img.copy())
            print(f"    Image {i}: alignment FAILED – using raw")

    return aligned


def select_reference(images: list[np.ndarray]) -> int:
    """Pick the image with the highest global focus score as reference."""
    scores = []
    for img in images:
        gray = to_gray_float(img)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        scores.append(float(np.mean(lap * lap)))
    return int(np.argmax(scores))


# ==============================================================
#  3.  FOCUS MEASURE COMPUTATION
# ==============================================================
def modified_laplacian(gray: np.ndarray) -> np.ndarray:
    """
    Modified Laplacian focus measure (Nayar & Nakagawa, 1994).
    Sum of absolute second derivatives in x and y.
    """
    ksize = FOCUS_KERNEL_SIZE
    Lxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=ksize)
    Lyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=ksize)
    ml = np.abs(Lxx) + np.abs(Lyy)
    return ml


def tenengrad(gray: np.ndarray) -> np.ndarray:
    """
    Tenengrad focus measure – squared gradient magnitude.
    """
    ksize = FOCUS_KERNEL_SIZE
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return gx * gx + gy * gy


def compute_focus_maps(images: list[np.ndarray]) -> np.ndarray:
    """
    Compute per-pixel focus score for every image in the stack.
    Combines Modified Laplacian and Tenengrad for robustness.

    Returns: float32 array of shape (N, H, W) – higher = sharper.
    """
    n = len(images)
    h, w = images[0].shape[:2]
    focus_volume = np.zeros((n, h, w), dtype=np.float32)

    for i, img in enumerate(images):
        gray = to_gray_float(img)

        ml = modified_laplacian(gray)
        tg = tenengrad(gray)

        # Local variance: Var = E[X²] − E[X]²
        # Tiny contribution (5%) acts as tiebreaker in smooth/low-texture regions
        # where Laplacian and Tenengrad both return near-zero
        mean = cv2.GaussianBlur(gray, (9, 9), 0)
        lv = cv2.GaussianBlur(gray * gray, (9, 9), 0) - mean * mean
        lv = np.maximum(lv, 0.0)

        # Combine: weighted sum (all normalised to [0,1] first)
        ml_n = cv2.normalize(ml, None, 0, 1, cv2.NORM_MINMAX)
        tg_n = cv2.normalize(tg, None, 0, 1, cv2.NORM_MINMAX)
        lv_n = cv2.normalize(lv, None, 0, 1, cv2.NORM_MINMAX)
        combined = 0.48 * ml_n + 0.48 * tg_n + 0.04 * lv_n

        # Spatial smoothing to suppress noise in flat regions
        combined = cv2.GaussianBlur(
            combined,
            (FOCUS_GAUSS_KSIZE, FOCUS_GAUSS_KSIZE),
            sigmaX=FOCUS_GAUSS_SIGMA,
        )
        # Gamma enhancement: increases contrast between sharp and blurred scores
        # in low-texture regions where raw differences are very small
        combined = np.power(combined, 1.4)
        focus_volume[i] = combined

    return focus_volume


# ==============================================================
#  4.  DECISION MAP GENERATION
# ==============================================================
def build_confidence_maps(focus_volume: np.ndarray) -> np.ndarray:
    """
    Convert raw focus scores into per-image soft confidence weights
    that sum to 1 at every pixel (softmax-style).

    Applies spatial consistency via morphological filtering and
    Gaussian smoothing to eliminate abrupt region switching (ghosting).

    Returns: float32 array of shape (N, H, W) with weights in [0, 1].
    """
    n, h, w = focus_volume.shape

    # ----------------------------------------------------------
    # Step 1: Per-pixel argmax → hard decision index map
    # ----------------------------------------------------------
    hard_index = np.argmax(focus_volume, axis=0).astype(np.uint8)

    # ----------------------------------------------------------
    # Step 2: Spatial consistency – morphological open then close
    #         on each binary channel to remove tiny islands
    # ----------------------------------------------------------
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (DECISION_MORPH_KSIZE, DECISION_MORPH_KSIZE),
    )

    cleaned_masks = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        binary = (hard_index == i).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned_masks[i] = binary.astype(np.float32) / 255.0

    # ----------------------------------------------------------
    # Step 3: Re-normalise so masks sum to 1 at every pixel
    # ----------------------------------------------------------
    mask_sum = cleaned_masks.sum(axis=0, keepdims=True) + 1e-8
    cleaned_masks /= mask_sum

    # ----------------------------------------------------------
    # Step 4: Heavy Gaussian smoothing → soft transitions
    # ----------------------------------------------------------
    for i in range(n):
        cleaned_masks[i] = cv2.GaussianBlur(
            cleaned_masks[i],
            (DECISION_SMOOTH_KSIZE, DECISION_SMOOTH_KSIZE),
            sigmaX=DECISION_SMOOTH_SIGMA,
        )

    # Final re-normalisation
    mask_sum = cleaned_masks.sum(axis=0, keepdims=True) + 1e-8
    cleaned_masks /= mask_sum

    return cleaned_masks


# ==============================================================
#  5a.  FUSION – STRATEGY A: LAPLACIAN PYRAMID
# ==============================================================
def _build_laplacian_pyramid(img: np.ndarray, levels: int):
    """Build Gaussian and Laplacian pyramids for *img*."""
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        down = cv2.pyrDown(gp[-1])
        gp.append(down)

    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lap = gp[i] - up
        lp.append(lap)
    lp.append(gp[-1])  # lowest-resolution residual

    return lp


def _build_gaussian_pyramid(mask: np.ndarray, levels: int):
    """Build a Gaussian pyramid for a single-channel weight mask."""
    gp = [mask]
    for _ in range(levels):
        down = cv2.pyrDown(gp[-1])
        gp.append(down)
    return gp


def _reconstruct_from_laplacian(lp):
    """Reconstruct image from its Laplacian pyramid."""
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        up = cv2.pyrUp(img, dstsize=(lp[i].shape[1], lp[i].shape[0]))
        img = up + lp[i]
    return img


def fuse_laplacian_pyramid(
    images: list[np.ndarray],
    weights: np.ndarray,
    levels: int = PYRAMID_LEVELS,
) -> np.ndarray:
    """
    Multi-band (Laplacian Pyramid) fusion.

    For each image, build a Laplacian pyramid of the image and a
    Gaussian pyramid of its weight mask.  At every pyramid level,
    blend using the weights.  Reconstruct the final image from the
    blended pyramid.

    This ensures:
      • High-frequency detail (edges) is blended only over narrow seam
      • Low-frequency colour/brightness is blended over wide area
      → seamless, ghost-free result
    """
    n = len(images)
    h, w = images[0].shape[:2]

    # Ensure dimensions are divisible by 2^levels
    factor = 2 ** levels
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor
    if new_h != h or new_w != w:
        images = [img[:new_h, :new_w] for img in images]
        weights = weights[:, :new_h, :new_w]

    # Build pyramids
    img_pyramids = [_build_laplacian_pyramid(img, levels) for img in images]
    # Weight masks: expand to 3-channel for multiplication
    wt_pyramids = [_build_gaussian_pyramid(w_map, levels) for w_map in weights]

    # Blend at each level
    blended_pyramid = []
    for lvl in range(levels + 1):
        blended = np.zeros_like(img_pyramids[0][lvl])
        for i in range(n):
            w3 = cv2.merge([wt_pyramids[i][lvl]] * 3)
            blended += img_pyramids[i][lvl] * w3
        blended_pyramid.append(blended)

    # Reconstruct
    fused = _reconstruct_from_laplacian(blended_pyramid)
    return np.clip(fused, 0, 255).astype(np.uint8)


# ==============================================================
#  5b.  FUSION – STRATEGY B: WEIGHTED AVERAGE (Baseline)
# ==============================================================
def fuse_weighted_average(
    images: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """
    Direct pixel-wise weighted average using the confidence maps.
    Simpler than pyramid blending but more prone to halo artifacts
    because it mixes frequency bands indiscriminately.
    """
    n = len(images)
    h, w = images[0].shape[:2]
    fused = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(n):
        w3 = weights[i][:h, :w][..., None]  # (H, W, 1)
        fused += images[i].astype(np.float64) * w3

    return np.clip(fused, 0, 255).astype(np.uint8)


# ==============================================================
#  6.  GHOST ARTIFACT REMOVAL (Additional pass)
# ==============================================================
def suppress_ghosts(
    fused: np.ndarray,
    images: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """
    Post-fusion ghost suppression.

    • Identify pixels where no single image dominates (max weight < 0.55).
      These are "uncertain" transition pixels prone to ghosting.
    • In uncertain regions, replace the blended pixel with the pixel
      from the single sharpest source → eliminates double-edge ghosts.
    """
    h, w = fused.shape[:2]

    # Crop weights and images to match the (possibly trimmed) fused size
    weights_cropped = weights[:, :h, :w]

    max_weight = np.max(weights_cropped, axis=0)   # (H, W)
    best_idx = np.argmax(weights_cropped, axis=0)  # (H, W)

    # Threshold: pixels with low confidence are ghost-prone
    ghost_mask = (max_weight < 0.55).astype(np.float32)

    # Smooth the ghost mask so replacement blends in naturally
    ghost_mask = cv2.GaussianBlur(ghost_mask, (21, 21), sigmaX=5.0)

    # Build replacement image from best-index per pixel (memory-efficient)
    replacement = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(len(images)):
        mask_i = (best_idx == i)  # boolean (H, W)
        if not mask_i.any():
            continue
        src = images[i][:h, :w].astype(np.float32)
        for c in range(3):
            replacement[:, :, c][mask_i] = src[:, :, c][mask_i]

    ghost_mask_3 = ghost_mask[..., None]  # (H, W, 1)
    corrected = (
        fused.astype(np.float32) * (1.0 - ghost_mask_3)
        + replacement * ghost_mask_3
    )

    return np.clip(corrected, 0, 255).astype(np.uint8)


# ==============================================================
#  7.  POST-PROCESSING
# ==============================================================
def unsharp_mask(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Apply unsharp-mask sharpening."""
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement in LAB colour space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
    )
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def postprocess(img: np.ndarray) -> np.ndarray:
    """Full post-processing chain."""
    img = unsharp_mask(img, SHARPEN_SIGMA, SHARPEN_AMOUNT)
    img = apply_clahe(img)
    return img


# ==============================================================
#  8.  VISUALISATION
# ==============================================================
def save_visualisation(
    out_dir: str,
    set_name: str,
    input_images: list[np.ndarray],
    input_names: list[str],
    focus_volume: np.ndarray,
    confidence_maps: np.ndarray,
    fused_pyramid: np.ndarray,
    fused_wavg: np.ndarray,
):
    """Save diagnostic visualisations as PNG files."""
    vis_dir = os.path.join(out_dir, f"{set_name}_visualisations")
    os.makedirs(vis_dir, exist_ok=True)

    # --- Input thumbnails ---
    n = len(input_images)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if idx < n:
            thumb = cv2.resize(input_images[idx], (400, 300))
            ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            ax.set_title(input_names[idx][:20], fontsize=7)
        ax.axis("off")
    fig.suptitle(f"{set_name} – Input Images", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "01_inputs.png"), dpi=120)
    plt.close(fig)

    # --- Focus maps ---
    rows_fm = min(n, 6)
    fig, axes = plt.subplots(1, rows_fm, figsize=(4 * rows_fm, 4))
    if rows_fm == 1:
        axes = [axes]
    for idx in range(rows_fm):
        fm = focus_volume[idx]
        fm_vis = cv2.normalize(fm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fm_vis = cv2.resize(fm_vis, (400, 300))
        axes[idx].imshow(fm_vis, cmap="hot")
        axes[idx].set_title(f"Focus {idx}", fontsize=8)
        axes[idx].axis("off")
    fig.suptitle(f"{set_name} – Focus Measure Maps", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "02_focus_maps.png"), dpi=120)
    plt.close(fig)

    # --- Confidence / decision maps ---
    rows_dm = min(n, 6)
    fig, axes = plt.subplots(1, rows_dm, figsize=(4 * rows_dm, 4))
    if rows_dm == 1:
        axes = [axes]
    for idx in range(rows_dm):
        dm = confidence_maps[idx]
        dm_vis = (dm * 255).astype(np.uint8)
        dm_vis = cv2.resize(dm_vis, (400, 300))
        axes[idx].imshow(dm_vis, cmap="viridis")
        axes[idx].set_title(f"Weight {idx}", fontsize=8)
        axes[idx].axis("off")
    fig.suptitle(f"{set_name} – Confidence Maps (Decision Weights)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "03_confidence_maps.png"), dpi=120)
    plt.close(fig)

    # --- Composite decision map (argmax overlay) ---
    hard_idx = np.argmax(confidence_maps, axis=0).astype(np.float32)
    hard_vis = cv2.normalize(hard_idx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hard_vis = cv2.applyColorMap(hard_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(vis_dir, "04_decision_map.png"), hard_vis)

    # --- Side-by-side comparison ---
    h = min(fused_pyramid.shape[0], fused_wavg.shape[0])
    w = min(fused_pyramid.shape[1], fused_wavg.shape[1])
    comparison = np.hstack([fused_pyramid[:h, :w], fused_wavg[:h, :w]])
    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Laplacian Pyramid", (30, 60), font, 2.0, (0, 255, 0), 3)
    cv2.putText(comparison, "Weighted Average", (w + 30, 60), font, 2.0, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(vis_dir, "05_comparison.png"), comparison)

    print(f"    Visualisations saved to: {vis_dir}")


# ==============================================================
#  MAIN PIPELINE – process one set (folder of images)
# ==============================================================
def process_set(
    input_folder: str,
    output_dir: str,
    set_name: str,
):
    """Run the full focus stacking pipeline on one set of images."""
    print(f"\n{'='*60}")
    print(f"  Processing: {set_name}")
    print(f"  Input:      {input_folder}")
    print(f"{'='*60}")

    # ── 1. Load ──
    t0 = time.time()
    pairs = load_images(input_folder)
    if len(pairs) < 2:
        print("  [SKIP] Fewer than 2 images – nothing to fuse.")
        return
    names = [p[0] for p in pairs]
    raw_images = [p[1] for p in pairs]
    print(f"  Loaded {len(raw_images)} images  ({raw_images[0].shape})  [{time.time()-t0:.1f}s]")

    # ── 2. Preprocessing ──
    t0 = time.time()
    images = [normalize_intensity(img) for img in raw_images]
    print(f"  Intensity normalised  [{time.time()-t0:.1f}s]")

    # ── 3. Registration ──
    t0 = time.time()
    ref_idx = select_reference(images)
    print(f"  Reference image: index {ref_idx}  ({names[ref_idx]})")
    aligned = align_stack(images, ref_idx)
    print(f"  Alignment complete  [{time.time()-t0:.1f}s]")

    # ── 4. Focus measure ──
    t0 = time.time()
    focus_volume = compute_focus_maps(aligned)
    print(f"  Focus maps computed  [{time.time()-t0:.1f}s]")

    # ── 5. Decision maps ──
    t0 = time.time()
    confidence = build_confidence_maps(focus_volume)
    print(f"  Confidence maps built  [{time.time()-t0:.1f}s]")

    # ── 6a. Fusion – Laplacian Pyramid ──
    t0 = time.time()
    fused_pyr = fuse_laplacian_pyramid(aligned, confidence, levels=PYRAMID_LEVELS)
    fused_pyr = suppress_ghosts(fused_pyr, aligned, confidence)
    fused_pyr = postprocess(fused_pyr)
    print(f"  Strategy A (Laplacian Pyramid) fused  [{time.time()-t0:.1f}s]")

    # ── 6b. Fusion – Weighted Average ──
    t0 = time.time()
    fused_wavg = fuse_weighted_average(aligned, confidence)
    fused_wavg = suppress_ghosts(fused_wavg, aligned, confidence)
    fused_wavg = postprocess(fused_wavg)
    print(f"  Strategy B (Weighted Average) fused  [{time.time()-t0:.1f}s]")

    # ── 7. Save outputs ──
    os.makedirs(output_dir, exist_ok=True)
    pyr_path = os.path.join(output_dir, f"{set_name}_fused_laplacian_pyramid.png")
    avg_path = os.path.join(output_dir, f"{set_name}_fused_weighted_average.png")
    cv2.imwrite(pyr_path, fused_pyr)
    cv2.imwrite(avg_path, fused_wavg)
    print(f"  Saved: {pyr_path}")
    print(f"  Saved: {avg_path}")

    # ── 8. Visualisations ──
    save_visualisation(
        output_dir, set_name,
        aligned, names,
        focus_volume, confidence,
        fused_pyr, fused_wavg,
    )


# ==============================================================
#  BATCH RUNNER – iterate over final_Dataset
# ==============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "final_Dataset")
    output_dir = os.path.join(dataset_dir, "Fused_Outputs")

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Discover set sub-folders
    set_folders = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and d.lower().startswith("set")
    ])

    if not set_folders:
        print("[ERROR] No 'Set …' sub-folders found in final_Dataset.")
        sys.exit(1)

    print(f"Found {len(set_folders)} image sets: {set_folders}")
    overall_t0 = time.time()

    for folder_name in set_folders:
        set_path = os.path.join(dataset_dir, folder_name)
        safe_name = folder_name.replace(" ", "_")
        process_set(set_path, output_dir, safe_name)

    total = time.time() - overall_t0
    print(f"\n{'='*60}")
    print(f"  ALL DONE  –  Total time: {total:.1f}s")
    print(f"  Outputs in: {output_dir}")
    print(f"{'='*60}")

    # ── Comparison summary ──
    print("""
╔══════════════════════════════════════════════════════════════╗
║           FUSION STRATEGY COMPARISON SUMMARY                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Strategy A: Laplacian Pyramid Fusion                        ║
║  ─────────────────────────────────────                       ║
║  • Decomposes each image into frequency bands                ║
║  • Blends high-frequency detail over very narrow seams       ║
║  • Blends low-frequency colour over wide transitions         ║
║  → BEST for eliminating ghosting because misaligned edges    ║
║    only affect the high-frequency band and are blended over  ║
║    a tiny spatial region.                                    ║
║                                                              ║
║  Strategy B: Weighted Average Fusion                         ║
║  ─────────────────────────────────────                       ║
║  • Directly averages pixel intensities with focus weights    ║
║  • Simple and fast                                           ║
║  → WORSE for ghosting because it mixes ALL frequencies      ║
║    equally; a misaligned sharp edge from one image bleeds    ║
║    into the other as a translucent "ghost".                  ║
║                                                              ║
║  WHY GHOSTING IS REDUCED:                                    ║
║  1. SIFT + Homography alignment corrects scale changes       ║
║     from focus breathing, so edges align precisely.           ║
║  2. Morphological cleaning removes tiny noisy islands in     ║
║     the decision map that would cause pixel flickering.       ║
║  3. Laplacian Pyramid blending only merges sharp detail      ║
║     over a very narrow band, so any residual misalignment    ║
║     does NOT produce a visible double-edge.                  ║
║  4. Ghost suppression pass replaces uncertain (low-weight)   ║
║     transition pixels with the single best source pixel.     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()