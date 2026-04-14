import os
import cv2
import numpy as np
from tqdm import tqdm

# =======================
# CONFIG
# =======================
ROOT_FOLDER = "Images Dataset"
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "FINAL_FUSED_OUTPUT")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

WORK_WIDTH = 1200
MASK_BLUR = 41

ECC_MODE = cv2.MOTION_TRANSLATION
ECC_ITERS = 1500
ECC_EPS = 1e-6


# =======================
# UTILS
# =======================
def resize_keep_aspect(img, target_w):
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)), cv2.INTER_AREA)


def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def ecc_align(ref, mov):
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mov_g = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)

    try:
        _, warp = cv2.findTransformECC(ref_g, mov_g, warp, ECC_MODE, criteria)
    except cv2.error:
        return mov, False

    h, w = ref.shape[:2]
    aligned = cv2.warpAffine(
        mov, warp, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned, True


def tenengrad_focus_map(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    fm = gx * gx + gy * gy
    return cv2.normalize(fm, None, 0, 1.0, cv2.NORM_MINMAX)


def fuse(imgA, imgB):
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    fmA = tenengrad_focus_map(gA)
    fmB = tenengrad_focus_map(gB)

    mask = (fmA > fmB).astype(np.float32)
    if MASK_BLUR % 2 == 0:
        k = MASK_BLUR + 1
    else:
        k = MASK_BLUR

    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0, 1)

    mask3 = cv2.merge([mask, mask, mask])
    fused = imgA.astype(np.float32) * mask3 + imgB.astype(np.float32) * (1 - mask3)
    return np.clip(fused, 0, 255).astype(np.uint8)


# =======================
# MAIN
# =======================
def main():
    object_folders = [
        d for d in os.listdir(ROOT_FOLDER)
        if os.path.isdir(os.path.join(ROOT_FOLDER, d)) and d.startswith("obj")
    ]

    print(f"Found {len(object_folders)} object groups")

    for obj in tqdm(object_folders, desc="Processing objects"):
        obj_path = os.path.join(ROOT_FOLDER, obj)
        files = sorted([f for f in os.listdir(obj_path) if f.lower().endswith(".tif")])

        if len(files) < 2:
            continue

        images = []
        scores = []

        for f in files:
            img = cv2.imread(os.path.join(obj_path, f))
            if img is None:
                continue
            img_small = resize_keep_aspect(img, WORK_WIDTH)
            images.append(img)
            scores.append(blur_score(img_small))

        if len(images) < 2:
            continue

        # pick sharpest and blurriest
        sharp_idx = np.argmax(scores)
        blur_idx = np.argmin(scores)

        A = images[sharp_idx]
        B = images[blur_idx]

        # align
        B_aligned, ok = ecc_align(A, B)
        if not ok:
            continue

        fused = fuse(A, B_aligned)

        out_name = f"{obj}_FUSED.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), fused)

    print("✅ DONE. Outputs saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
