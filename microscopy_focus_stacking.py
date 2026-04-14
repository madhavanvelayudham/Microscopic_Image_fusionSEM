import argparse
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an all-in-focus image from a microscopy focus stack."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing the input images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Folder where fused outputs will be written.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="If > 0, split the input sequence into fixed-size stacks.",
    )
    parser.add_argument(
        "--auto-group-threshold",
        type=float,
        default=20.0,
        help=(
            "Mean grayscale difference threshold used to split consecutive images "
            "into new stacks when group-size is 0."
        ),
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=1200,
        help="Working width for alignment and grouping previews.",
    )
    parser.add_argument(
        "--ecc-motion",
        choices=["translation", "euclidean", "affine"],
        default="euclidean",
        help="Alignment model. For same microscope location, euclidean is a good default.",
    )
    parser.add_argument(
        "--ecc-iters",
        type=int,
        default=1000,
        help="Maximum ECC iterations.",
    )
    parser.add_argument(
        "--ecc-eps",
        type=float,
        default=1e-6,
        help="ECC convergence epsilon.",
    )
    parser.add_argument(
        "--focus-sigma",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma for the focus score map.",
    )
    parser.add_argument(
        "--weight-sharpness",
        type=float,
        default=12.0,
        help="Higher values make pixel selection more decisive.",
    )
    return parser.parse_args()


def list_images(folder):
    paths = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return paths


def resize_keep_aspect(image, target_width):
    height, width = image.shape[:2]
    if width <= target_width:
        return image.copy()
    scale = target_width / float(width)
    return cv2.resize(
        image,
        (target_width, int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def compute_global_focus_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.mean(lap * lap))


def select_reference_index(images):
    scores = [compute_global_focus_score(image) for image in images]
    return int(np.argmax(scores))


def motion_mode(name):
    modes = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    return modes[name]


def ecc_align(reference, moving, mode, iters, eps):
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mov_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    try:
        cv2.findTransformECC(ref_gray, mov_gray, warp, mode, criteria)
        aligned = cv2.warpAffine(
            moving,
            warp,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, True
    except cv2.error:
        return moving, False


def align_stack(images, preview_width, ecc_mode, ecc_iters, ecc_eps):
    ref_idx = select_reference_index(images)
    reference = images[ref_idx]
    preview_reference = resize_keep_aspect(reference, preview_width)

    aligned = []
    alignment_ok = []

    for idx, image in enumerate(images):
        if idx == ref_idx:
            aligned.append(image.copy())
            alignment_ok.append(True)
            continue

        preview_image = resize_keep_aspect(image, preview_width)
        _, preview_success = ecc_align(
            preview_reference,
            preview_image,
            ecc_mode,
            ecc_iters,
            ecc_eps,
        )

        if not preview_success:
            aligned.append(image.copy())
            alignment_ok.append(False)
            continue

        full_aligned, full_success = ecc_align(reference, image, ecc_mode, ecc_iters, ecc_eps)
        aligned.append(full_aligned if full_success else image.copy())
        alignment_ok.append(full_success)

    return aligned, ref_idx, alignment_ok


def focus_measure(image, sigma):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    score = np.abs(lap) + 0.5 * grad_mag
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return score


def build_weight_volume(images, sigma, sharpness):
    focus_maps = np.stack([focus_measure(image, sigma) for image in images], axis=0)

    focus_maps -= focus_maps.max(axis=0, keepdims=True)
    weights = np.exp(focus_maps * sharpness)
    weights_sum = np.sum(weights, axis=0, keepdims=True) + 1e-8
    return weights / weights_sum


def fuse_stack(images, sigma, sharpness):
    weights = build_weight_volume(images, sigma, sharpness)
    stack = np.stack(images, axis=0).astype(np.float32)
    fused = np.sum(stack * weights[..., None], axis=0)
    best_index = np.argmax(weights, axis=0).astype(np.uint8)
    return np.clip(fused, 0, 255).astype(np.uint8), best_index


def grayscale_preview(image, target_width):
    resized = resize_keep_aspect(image, target_width)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def auto_group(paths, preview_width, threshold):
    if not paths:
        return []

    groups = [[paths[0]]]
    prev = cv2.imread(str(paths[0]))
    if prev is None:
        raise ValueError(f"Unable to read image: {paths[0]}")
    prev_gray = grayscale_preview(prev, preview_width)

    for path in paths[1:]:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        gray = grayscale_preview(image, preview_width)

        diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
        if diff > threshold:
            groups.append([path])
        else:
            groups[-1].append(path)

        prev_gray = gray

    return groups


def fixed_groups(paths, group_size):
    return [paths[idx:idx + group_size] for idx in range(0, len(paths), group_size)]


def save_index_map(index_map, output_path, levels):
    if levels <= 1:
        view = np.zeros_like(index_map, dtype=np.uint8)
    else:
        view = np.round(index_map.astype(np.float32) * (255.0 / (levels - 1))).astype(np.uint8)
    cv2.imwrite(str(output_path), view)


def process_group(group_paths, output_dir, args, group_id):
    images = []
    used_paths = []

    for path in group_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] Skipping unreadable file: {path}")
            continue
        images.append(image)
        used_paths.append(path)

    if len(images) < 2:
        print(f"[SKIP] Group {group_id:03d} has fewer than two readable images.")
        return

    aligned, ref_idx, alignment_ok = align_stack(
        images,
        preview_width=args.preview_width,
        ecc_mode=motion_mode(args.ecc_motion),
        ecc_iters=args.ecc_iters,
        ecc_eps=args.ecc_eps,
    )
    fused, index_map = fuse_stack(
        aligned,
        sigma=args.focus_sigma,
        sharpness=args.weight_sharpness,
    )

    stem = used_paths[0].stem if len(used_paths) == 1 else f"group_{group_id:03d}_{used_paths[0].stem}"
    fused_path = output_dir / f"{stem}_all_in_focus.png"
    map_path = output_dir / f"{stem}_focus_map.png"

    cv2.imwrite(str(fused_path), fused)
    save_index_map(index_map, map_path, len(aligned))

    ok_count = sum(1 for item in alignment_ok if item)
    print(
        f"[OK] Group {group_id:03d}: {len(aligned)} images fused | "
        f"reference={used_paths[ref_idx].name} | alignments={ok_count}/{len(aligned)}"
    )


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_images(input_dir)
    if len(paths) < 2:
        raise SystemExit("Need at least two images in the input folder.")

    if args.group_size > 0:
        groups = fixed_groups(paths, args.group_size)
        grouping_mode = f"fixed size {args.group_size}"
    else:
        groups = auto_group(paths, args.preview_width, args.auto_group_threshold)
        grouping_mode = f"auto threshold {args.auto_group_threshold}"

    valid_groups = [group for group in groups if len(group) >= 2]
    print(
        f"Found {len(paths)} images and {len(valid_groups)} processable groups using {grouping_mode}."
    )

    for group_id, group in enumerate(valid_groups, start=1):
        process_group(group, output_dir, args, group_id)

    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
