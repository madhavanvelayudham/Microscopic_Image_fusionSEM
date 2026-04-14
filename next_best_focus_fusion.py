import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===========================
# CONFIG (EDIT THIS PATH)
# ===========================
INPUT_FOLDER = r"D:\Intern PSG\Images Dataset"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "FUSED_RESULTS_FINAL")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Search range for best pair
LOOKAHEAD = 6  # checks i+1 to i+6

# Fast processing resize for pairing/alignment/mask
WORK_WIDTH = 1200

# Pair selection thresholds
MAX_SIMILARITY = 28.0        # lower = stricter (try 20~35)
MIN_FOCUS_DIFF = 8.0         # must have some focus difference

# ECC settings
ECC_MODE = cv2.MOTION_TRANSLATION   # safest (no crazy warp)
ECC_ITERS = 1500
ECC_EPS = 1e-6

# Fusion mask smoothing
MASK_BLUR = 41


# ===========================
# UTILS
# ===========================
def resize_keep_aspect(img, target_w):
    h, w = img.shape[:2]
    if w <= target_w:
        return img, 1.0
    scale = target_w / w
    new_w = target_w
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def similarity_score(imgA, imgB):
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gA, gB)
    return float(diff.mean())


# ===========================
# ECC ALIGNMENT (SAFE)
# ===========================
def ecc_align(ref, mov):
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mov_g = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if ECC_MODE == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)

    try:
        _, warp = cv2.findTransformECC(ref_g, mov_g, warp, ECC_MODE, criteria)
    except cv2.error:
        return mov, False

    h, w = ref.shape[:2]

    if ECC_MODE == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(mov, warp, (w, h),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(mov, warp, (w, h),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned, True


# ===========================
# FOCUS MAP + MASK
# ===========================
def tenengrad_focus_map(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    fm = gx * gx + gy * gy
    fm = cv2.normalize(fm, None, 0, 1.0, cv2.NORM_MINMAX)
    return fm


def build_soft_mask(imgA, imgB):
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    fmA = tenengrad_focus_map(gA)
    fmB = tenengrad_focus_map(gB)

    mask = (fmA > fmB).astype(np.float32)

    k = MASK_BLUR
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0, 1)


def fuse(imgA, imgB, mask):
    m3 = cv2.merge([mask, mask, mask])
    out = imgA.astype(np.float32) * m3 + imgB.astype(np.float32) * (1 - m3)
    return np.clip(out, 0, 255).astype(np.uint8)


# ===========================
# BEST PAIR PICKER (MAX OUTPUT)
# ===========================
def find_best_pair(images_small, scores, i):
    """
    Pick best j in [i+1 .. i+LOOKAHEAD] using:
    - similarity low
    - focus difference high
    """
    best = None
    best_sim = 1e9

    for j in range(i + 1, min(len(images_small), i + 1 + LOOKAHEAD)):
        sim = similarity_score(images_small[i], images_small[j])
        focus_diff = abs(scores[i] - scores[j])

        if sim <= MAX_SIMILARITY and focus_diff >= MIN_FOCUS_DIFF:
            if sim < best_sim:
                best_sim = sim
                best = j

    return best, best_sim


# ===========================
# MAIN
# ===========================
def main():
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".tif")])

    if len(files) < 2:
        print("❌ Not enough images in folder.")
        return

    print("✅ Loading images (small preview + focus score)...")

    names = []
    full_imgs = []
    small_imgs = []
    focus_scores = []

    for f in tqdm(files):
        path = os.path.join(INPUT_FOLDER, f)
        img = cv2.imread(path)

        if img is None:
            continue

        small, _ = resize_keep_aspect(img, WORK_WIDTH)

        names.append(f)
        full_imgs.append(img)
        small_imgs.append(small)
        focus_scores.append(blur_score(small))

    print(f"✅ Loaded {len(names)} images")

    used = set()
    logs = []
    fused_count = 0

    print("✅ Finding best pairs + fusing...")

    for i in tqdm(range(len(names) - 1)):
        if i in used:
            continue

        j, sim = find_best_pair(small_imgs, focus_scores, i)

        if j is None:
            logs.append([names[i], "-", "SKIP", "No suitable match", "-", "-"])
            continue

        if j in used:
            logs.append([names[i], names[j], "SKIP", "Match already used", sim, abs(focus_scores[i] - focus_scores[j])])
            continue

        # Alignment on small
        A_small = small_imgs[i]
        B_small = small_imgs[j]
        B_aligned_small, ok = ecc_align(A_small, B_small)

        if not ok:
            logs.append([names[i], names[j], "FAIL", "ECC alignment failed", sim, abs(focus_scores[i] - focus_scores[j])])
            continue

        # Build mask small -> upscale to full
        mask_small = build_soft_mask(A_small, B_aligned_small)

        A_full = full_imgs[i]
        B_full = full_imgs[j]

        # Align on full too (more accurate)
        B_aligned_full, ok2 = ecc_align(A_full, B_full)
        if not ok2:
            logs.append([names[i], names[j], "FAIL", "ECC full alignment failed", sim, abs(focus_scores[i] - focus_scores[j])])
            continue

        mask_full = cv2.resize(mask_small, (A_full.shape[1], A_full.shape[0]), interpolation=cv2.INTER_LINEAR)

        fused = fuse(A_full, B_aligned_full, mask_full)

        out_name = f"FUSED_{fused_count:04d}_{names[i].replace('.tif','')}_{names[j].replace('.tif','')}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), fused)

        used.add(i)
        used.add(j)
        fused_count += 1

        logs.append([names[i], names[j], "OK", out_name, sim, abs(focus_scores[i] - focus_scores[j])])

    # Save logs
    df = pd.DataFrame(logs, columns=["Image_A", "Image_B", "Status", "Output/Reason", "SimilarityScore", "FocusDiff"])
    df.to_csv(os.path.join(OUTPUT_FOLDER, "fusion_log.csv"), index=False)

    print("\n✅ DONE")
    print(f"✅ Total fused outputs: {fused_count}")
    print(f"📁 Saved to: {OUTPUT_FOLDER}")
    print("📄 Log saved: fusion_log.csv")


if __name__ == "__main__":
    main()
