#!/usr/bin/env python3
"""
dirichlet_dataset.py (v2, FIXED) — builds FL_Run3_LabelSkew_Dirichlet.

FIX vs. v1: v1 padded every client's every class to a flat TARGET_PER_CLASS
via augmentation, which silently erased the Dirichlet-induced skew (every
client ended up ~class-balanced post-augmentation, identical in spirit to
Run1). v2 instead sets each client's per-class augmentation target
PROPORTIONAL to that client's own raw Dirichlet allocation, so the skew
survives augmentation. A max-multiplier cap avoids pathological reuse of a
handful of source images. Classes assigned zero raw images by Dirichlet stay
at zero -- that is a genuine, expected outcome of label skew, not an error.
"""

import os, random, json, shutil
import numpy as np
import cv2
import albumentations as A
from pathlib import Path
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASETS_ROOT  = "datasets"
FINAL_ROOT     = os.path.join(DATASETS_ROOT, "final_5_fold_pruned")
SOURCE_RUN_DIR = "FL_Run1_Uniform"
NEW_RUN_NAME   = "FL_Run3_LabelSkew_Dirichlet"

CLASSES      = ["Chickenpox", "Healthy", "Measles", "Monkeypox"]
NUM_CLASSES  = len(CLASSES)
SPLITS       = ["Train", "Valid", "Test"]
NUM_CLIENTS  = 5
NUM_FOLDS    = 5
FOLDS        = [f"Fold_{i}" for i in range(1, NUM_FOLDS + 1)]

DIRICHLET_ALPHA       = 0.3
TARGET_PER_CLIENT     = 1500     # approximate total training budget per client, as in Run1
MAX_AUG_MULTIPLIER    = 25       # cap on final/raw ratio -- avoids reusing 1-2 source images 100x+
MIN_RAW_PER_CLASS_CAP = 0        # 0 = allow genuine zero-sample classes per client (true Dirichlet skew).
                                  # Set >0 only if you want to force a floor and document that deviation.

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

AUG_PIPELINE = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
                        border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=1.0),
             A.MedianBlur(blur_limit=3, p=1.0),
             A.MotionBlur(blur_limit=3, p=1.0)], p=0.3),
    A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
             A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)], p=0.5),
    A.OneOf([A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
             A.ElasticTransform(alpha=1, sigma=5, p=1.0),
             A.OpticalDistortion(distort_limit=0.05, p=1.0)], p=0.25),
    A.CoarseDropout(max_holes=4, max_height=16, max_width=16, min_holes=1, fill_value=0, p=0.2),
    A.Resize(224, 224),
])

def list_images(directory):
    if not os.path.isdir(directory): return []
    return sorted(f for f in os.listdir(directory) if Path(f).suffix.lower() in IMG_EXTS)

def is_raw_image(fname):
    return not fname.startswith('aug_')

def reconstruct_fold_pool(fold_dir, split, cls):
    """Concatenate all 5 clients' folders for (split, cls) under
    SOURCE_RUN_DIR to exactly reconstruct the original fold-level pool."""
    paths = []
    for c in range(1, NUM_CLIENTS + 1):
        folder = os.path.join(fold_dir, SOURCE_RUN_DIR, f"Client_{c}", split, cls)
        for f in list_images(folder):
            if split == "Train" and not is_raw_image(f):
                continue
            paths.append(os.path.join(folder, f))
    return paths

def dirichlet_partition_counts(n, num_clients, alpha, rng, min_floor=0):
    """Draw a Dirichlet(alpha) proportion vector, convert to integer counts
    summing exactly to n via largest-remainder rounding. Optional min_floor
    redistributes to guarantee every client gets at least `min_floor` (only
    used if MIN_RAW_PER_CLASS_CAP > 0; disabled by default so the skew is
    genuine, including legitimate zero-sample clients)."""
    if n == 0:
        return [0] * num_clients
    props = rng.dirichlet([alpha] * num_clients)
    raw = props * n
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    frac_order = np.argsort(-(raw - counts))
    for i in range(remainder):
        counts[frac_order[i % num_clients]] += 1
    if min_floor > 0 and n >= min_floor * num_clients:
        counts = np.maximum(counts, min_floor)
        # renormalize back down to exactly n by trimming the largest entries
        while counts.sum() > n:
            idx = int(np.argmax(counts))
            if counts[idx] > min_floor:
                counts[idx] -= 1
            else:
                break
    return counts.tolist()

def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def create_client_dirs(base_dir):
    for c in range(1, NUM_CLIENTS + 1):
        for split in SPLITS:
            for cls in CLASSES:
                os.makedirs(os.path.join(base_dir, f"Client_{c}", split, cls), exist_ok=True)

def partition_uniform_from_pool(pool_by_class, split, base_dir, log):
    for cls in CLASSES:
        imgs = pool_by_class[cls][:]
        n = len(imgs)
        base_cnt = n // NUM_CLIENTS
        rem = n % NUM_CLIENTS
        start = 0
        for c in range(1, NUM_CLIENTS + 1):
            extra = 1 if c <= rem else 0
            chunk = imgs[start:start + base_cnt + extra]
            start += base_cnt + extra
            for i, src in enumerate(chunk):
                ext = Path(src).suffix
                fname = f"c{c}_{cls[:3]}_{split[:2]}_{i:04d}{ext}"
                copy_file(src, os.path.join(base_dir, f"Client_{c}", split, cls, fname))
            log.setdefault(f"Client_{c}", {}).setdefault(split, {})[cls] = len(chunk)

def partition_dirichlet_train(pool_by_class, base_dir, alpha, log, rng):
    """Train only -- genuine per-class Dirichlet label skew. Returns the RAW
    (pre-augmentation) per-client, per-class counts, which drive the
    augmentation targets below -- this is the source of truth for the skew."""
    dirichlet_record = {}
    raw_counts_by_client = {c: {} for c in range(1, NUM_CLIENTS + 1)}
    for cls in CLASSES:
        imgs = pool_by_class[cls][:]
        seed_local = int(rng.randint(0, 2**31 - 1))
        np.random.RandomState(seed_local).shuffle(imgs)
        n = len(imgs)
        counts = dirichlet_partition_counts(n, NUM_CLIENTS, alpha, rng, min_floor=MIN_RAW_PER_CLASS_CAP)
        dirichlet_record[cls] = {'n_total': n, 'client_counts': counts}
        start = 0
        for c_idx, cnt in enumerate(counts):
            c = c_idx + 1
            chunk = imgs[start:start + cnt]
            start += cnt
            for i, src in enumerate(chunk):
                ext = Path(src).suffix
                fname = f"c{c}_{cls[:3]}_tr_{i:04d}{ext}"
                copy_file(src, os.path.join(base_dir, f"Client_{c}", "Train", cls, fname))
            log.setdefault(f"Client_{c}", {}).setdefault("Train", {})[cls] = len(chunk)
            raw_counts_by_client[c][cls] = len(chunk)
    return dirichlet_record, raw_counts_by_client

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img

def compute_proportional_aug_targets(raw_counts_this_client, total_budget, max_multiplier):
    """THE FIX: per-class augmentation target scales with this client's own
    raw class mix, not a flat constant. A class with zero raw images gets
    target=0 (genuine skew preserved). A multiplier cap prevents blowing up
    a tiny raw pool (e.g. 2 images) by 100x+."""
    raw_total = sum(raw_counts_this_client.values())
    targets = {}
    if raw_total == 0:
        return {cls: 0 for cls in raw_counts_this_client}
    for cls, raw_n in raw_counts_this_client.items():
        if raw_n == 0:
            targets[cls] = 0
            continue
        proportional_target = int(round(total_budget * (raw_n / raw_total)))
        capped_target = min(proportional_target, raw_n * max_multiplier)
        targets[cls] = max(capped_target, raw_n)  # never shrink below what's already there
    return targets

def augment_client_train_proportional(client_dir, per_class_targets, aug_record):
    for cls, target_n in per_class_targets.items():
        folder = os.path.join(client_dir, "Train", cls)
        raw_images = list_images(folder)
        raw_count = len(raw_images)
        if raw_count == 0:
            aug_record[cls] = {"raw": 0, "final": 0, "generated": 0, "target": target_n}
            continue
        needed = max(0, target_n - raw_count)
        if needed > 0:
            src_paths = [os.path.join(folder, f) for f in raw_images]
            for i in range(needed):
                src = src_paths[i % raw_count]
                img = load_image(src)
                aug = AUG_PIPELINE(image=img)["image"]
                cv2.imwrite(os.path.join(folder, f"aug_{i:05d}.jpg"), aug)
        final_count = len(list_images(folder))
        aug_record[cls] = {"raw": raw_count, "final": final_count, "generated": needed,
                            "target": target_n, "multiplier": round(final_count / raw_count, 2)}

# ── Main loop ─────────────────────────────────────────────────────────────
rng = np.random.RandomState(SEED)
master_log = {}

for fold_name in FOLDS:
    fold_dir = os.path.join(FINAL_ROOT, fold_name)
    print(f"Processing {fold_name} ...")

    pool_train, pool_valid, pool_test = {}, {}, {}
    for cls in CLASSES:
        pool_train[cls] = reconstruct_fold_pool(fold_dir, "Train", cls)
        pool_valid[cls] = reconstruct_fold_pool(fold_dir, "Valid", cls)
        pool_test[cls]  = reconstruct_fold_pool(fold_dir, "Test", cls)
        print(f"  {cls}: raw-train={len(pool_train[cls])} valid={len(pool_valid[cls])} test={len(pool_test[cls])}")

    run3_dir = os.path.join(fold_dir, NEW_RUN_NAME)
    create_client_dirs(run3_dir)
    log = {}

    partition_uniform_from_pool(pool_valid, "Valid", run3_dir, log)
    partition_uniform_from_pool(pool_test,  "Test",  run3_dir, log)
    dirichlet_record, raw_counts_by_client = partition_dirichlet_train(pool_train, run3_dir, DIRICHLET_ALPHA, log, rng)

    aug_log = {}
    for c in range(1, NUM_CLIENTS + 1):
        ck = f"Client_{c}"
        targets = compute_proportional_aug_targets(
            raw_counts_by_client[c], TARGET_PER_CLIENT, MAX_AUG_MULTIPLIER)
        aug_log[ck] = {}
        augment_client_train_proportional(os.path.join(run3_dir, ck), targets, aug_log[ck])

    master_log[fold_name] = {
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "dirichlet_raw_assignment": dirichlet_record,
        "raw_partition_log": log,
        "augmentation_log": aug_log,
    }

with open("dirichlet_label_skew_distribution_log.json", "w") as f:
    json.dump(master_log, f, indent=2)

# ── Sanity check: verify Valid/Test parity, AND verify skew actually survived ──
print("\nSanity check ...")
issues = []
for fold in FOLDS:
    fold_dir = os.path.join(FINAL_ROOT, fold)
    for c in range(1, NUM_CLIENTS + 1):
        for split in SPLITS:
            for cls in CLASSES:
                run3_folder = os.path.join(fold_dir, NEW_RUN_NAME, f"Client_{c}", split, cls)
                n_run3 = len(list_images(run3_folder))
                if split in ("Valid", "Test"):
                    run1_folder = os.path.join(fold_dir, SOURCE_RUN_DIR, f"Client_{c}", split, cls)
                    n_run1 = len(list_images(run1_folder))
                    if n_run3 != n_run1:
                        issues.append(f"  [MISMATCH] {fold}/{NEW_RUN_NAME}/Client_{c}/{split}/{cls}: {n_run3} vs Run1's {n_run1}")
if issues:
    for i in issues: print(i)
else:
    print("  Valid/Test exactly match FL_Run1_Uniform.")

print("\nPost-augmentation class-proportion check (proof the skew survived):")
for fold in FOLDS:
    print(f"  {fold}:")
    for c in range(1, NUM_CLIENTS + 1):
        run3_dir = os.path.join(FINAL_ROOT, fold, NEW_RUN_NAME, f"Client_{c}", "Train")
        counts = {cls: len(list_images(os.path.join(run3_dir, cls))) for cls in CLASSES}
        total = sum(counts.values())
        props = {cls: round(counts[cls] / total, 3) if total else 0.0 for cls in CLASSES}
        print(f"    Client_{c}: counts={counts} total={total} proportions={props}")

print("\nDirichlet label-skew (FL_Run3) dataset generation complete (v2, skew-preserving).")