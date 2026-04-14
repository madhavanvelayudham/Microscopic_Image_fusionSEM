import os
import cv2
import numpy as np
from tqdm import tqdm

# ===========================
# CONFIG (EDIT THIS)
# ===========================
INPUT_FOLDER = r"D:\Intern PSG\Images Dataset"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "FUSED_RESULTS")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# If images are huge, downscale for faster alignment & focus mask computation
WORK_WIDTH = 1200  # increase if you want more accuracy, decrease for speed
SIMILARITY_THRESH = 35.0  # lower = stricter pairing (try 25~45)
MASK_BLUR = 41  # smoothing mask (odd number)


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
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def focus_score(gray):
    """Higher score => sharper image"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def similarity_score(imgA, imgB):
    """Lower score => more similar (same scene)"""
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gA, gB)
    return float(diff.mean())


# ===========================
# ALIGNMENT (ORB -> ECC fallback)
# ===========================
def orb_align(ref, mov, max_features=2500, good_match_percent=0.15):
    """
    Align mov to ref using ORB + Homography.
    Works well even with larger shifts.
    """
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(mov_gray, None)

    if des1 is None or des2 is None:
        return mov  # fallback

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 20:
        return mov

    matches = sorted(matches, key=lambda x: x.distance)

    num_good = int(len(matches) * good_match_percent)
    num_good = max(num_good, 15)
    matches = matches[:num_good]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    if H is None:
        return mov

    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(mov, H, (w, h), flags=cv2.INTER_LINEAR)
    return aligned


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
    """
    Create mask where pixels from A are chosen when A is sharper, else from B.
    """
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    fmA = tenengrad_focus_map(gA)
    fmB = tenengrad_focus_map(gB)

    mask = (fmA > fmB).astype(np.float32)

    # Smooth mask to avoid seams
    k = MASK_BLUR
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0, 1)
    return mask


def fuse_images(imgA, imgB, mask):
    """Fused = A*mask + B*(1-mask)"""
    mask3 = cv2.merge([mask, mask, mask])
    fused = imgA.astype(np.float32) * mask3 + imgB.astype(np.float32) * (1.0 - mask3)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused


# ===========================
# SMART PAIRING
# ===========================
def pick_best_pair(images, idx):
    """
    For image idx, choose best partner among idx+1 and idx+2
    based on similarity score.
    This solves cases where sequence contains 3 frames per scene.
    """
    base = images[idx]
    candidates = []

    if idx + 1 < len(images):
        candidates.append((idx + 1, images[idx + 1]))
    if idx + 2 < len(images):
        candidates.append((idx + 2, images[idx + 2]))

    best_j = None
    best_score = 1e9

    for j, img in candidates:
        s = similarity_score(base, img)
        if s < best_score:
            best_score = s
            best_j = j

    return best_j, best_score


# ===========================
# MAIN PROCESS
# ===========================
def main():
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".tif")])

    if len(files) < 2:
        print("❌ Not enough TIF images in folder.")
        return

    # Preload images (downscaled version for speed)
    print("✅ Loading images...")
    imgs_small = []
    imgs_full = []
    names = []

    for f in tqdm(files):
        path = os.path.join(INPUT_FOLDER, f)
        img = cv2.imread(path)

        if img is None:
            print("⚠️ Skipping unreadable:", f)
            continue

        small, _ = resize_keep_aspect(img, WORK_WIDTH)

        imgs_full.append(img)
        imgs_small.append(small)
        names.append(f)

    print(f"✅ Loaded {len(imgs_small)} images successfully")

    used = set()
    fused_count = 0

    print("✅ Processing fusion...")

    i = 0
    while i < len(imgs_small) - 1:
        if i in used:
            i += 1
            continue

        j, sim = pick_best_pair(imgs_small, i)

        if j is None:
            i += 1
            continue

        # Reject if too different
        if sim > SIMILARITY_THRESH:
            i += 1
            continue

        # Mark used
        used.add(i)
        used.add(j)

        # Get full images
        A_full = imgs_full[i]
        B_full = imgs_full[j]

        # Resize full images for alignment + mask generation speed
        A_small, scaleA = resize_keep_aspect(A_full, WORK_WIDTH)
        B_small, scaleB = resize_keep_aspect(B_full, WORK_WIDTH)

        # Align (on small images)
        B_aligned_small = orb_align(A_small, B_small)

        # Create mask on small
        mask_small = build_soft_mask(A_small, B_aligned_small)

        # Upscale mask to full size
        mask_full = cv2.resize(mask_small, (A_full.shape[1], A_full.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Align full B (using same ORB on full might be slow, so we align small then warp full with feature alignment again)
        # For best quality, align full directly one time:
        B_aligned_full = orb_align(A_full, B_full)

        # Fuse full resolution
        fused = fuse_images(A_full, B_aligned_full, mask_full)

        # Save output
        out_name = f"FUSED_{fused_count:04d}_{names[i].replace('.tif','')}_{names[j].replace('.tif','')}.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        cv2.imwrite(out_path, fused)
        fused_count += 1

        i += 1

    print(f"✅ DONE! Total fused outputs saved: {fused_count}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
