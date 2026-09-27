#!/usr/bin/env python3
"""
dirichlet_dataset.py — builds FL_Run3_LabelSkew_Dirichlet, a genuine label-
distribution non-IID partition, addressing Peer-Review Major Comment 2.

WHY THIS APPROACH:
Re-deriving the original fold-level Train/Valid/Test pools by replaying
dataset_creation.md's RNG sequence is fragile (any extra random call anywhere
in that pipeline before this point would silently desync the shuffle and give
different Test/Valid images than Run1/Run2, breaking comparability). Instead,
this script RECONSTRUCTS the exact original pools by reading the files already
materialized under FL_Run1_Uniform: concatenating all 5 clients' folders for
a given (split, class) gives back precisely the original fold-level pool,
since partition_uniform() never drops or duplicates an image. This guarantees
Run3's Valid/Test sets are byte-for-byte identical to Run1/Run2's.

DESIGN CHOICES (stated explicitly):
- Valid/Test: kept UNIFORM across clients (same as Run1) -- evaluation in the
  main pipeline is always on the POOLED test/val set across all clients
  anyway, so this isolates the manipulated variable to Train only.
- Train: genuine per-class Dirichlet(alpha) label skew (Hsu et al., 2019),
  drawn independently per class, so clients differ in CLASS PROPORTION, not
  just quantity.
- Total per-client Train volume after augmentation is fixed at 1500 (same as
  Run1), so this experiment does not reintroduce quantity skew as a
  confound -- only label-distribution skew is being tested.
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
SOURCE_RUN_DIR = "FL_Run1_Uniform"          # canonical source for reconstructing fold pools
NEW_RUN_NAME   = "FL_Run3_LabelSkew_Dirichlet"

CLASSES      = ["Chickenpox", "Healthy", "Measles", "Monkeypox"]
NUM_CLASSES  = len(CLASSES)
SPLITS       = ["Train", "Valid", "Test"]
NUM_CLIENTS  = 5
NUM_FOLDS    = 5
FOLDS        = [f"Fold_{i}" for i in range(1, NUM_FOLDS + 1)]

DIRICHLET_ALPHA    = 0.3          # moderate label-distribution skew
TARGET_PER_CLIENT  = 1500         # same total budget as Run1 -- isolates label skew
TARGET_PER_CLASS   = TARGET_PER_CLIENT // NUM_CLASSES

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
    # augmented copies are named "aug_00000.jpg" etc.; raw copies are
    # "c{client}_{cls}_{split}_{idx}{ext}" -- see partition_uniform/heterogeneous
    return not fname.startswith('aug_')

def reconstruct_fold_pool(fold_dir, split, cls):
    """Concatenate all 5 clients' folders for (split, cls) under
    SOURCE_RUN_DIR to exactly reconstruct the original fold-level pool.
    For Train, only RAW (pre-augmentation) copies are kept."""
    paths = []
    for c in range(1, NUM_CLIENTS + 1):
        folder = os.path.join(fold_dir, SOURCE_RUN_DIR, f"Client_{c}", split, cls)
        for f in list_images(folder):
            if split == "Train" and not is_raw_image(f):
                continue
            paths.append(os.path.join(folder, f))
    return paths

def dirichlet_partition_counts(n, num_clients, alpha, rng):
    """Draw a Dirichlet(alpha) proportion vector, convert to integer counts
    summing exactly to n via largest-remainder rounding."""
    if n == 0:
        return [0] * num_clients
    props = rng.dirichlet([alpha] * num_clients)
    raw = props * n
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    frac_order = np.argsort(-(raw - counts))
    for i in range(remainder):
        counts[frac_order[i % num_clients]] += 1
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
    """Valid/Test only -- kept uniform, identical counts to Run1."""
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
    """Train only -- genuine per-class Dirichlet label skew."""
    dirichlet_record = {}
    for cls in CLASSES:
        imgs = pool_by_class[cls][:]
        seed_local = int(rng.randint(0, 2**31 - 1))
        np.random.RandomState(seed_local).shuffle(imgs)
        n = len(imgs)
        counts = dirichlet_partition_counts(n, NUM_CLIENTS, alpha, rng)
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
    return dirichlet_record

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img

def augment_client_train(client_dir, target_per_class, aug_record):
    for cls in CLASSES:
        folder = os.path.join(client_dir, "Train", cls)
        raw_images = list_images(folder)
        raw_count = len(raw_images)
        if raw_count == 0:
            aug_record[cls] = {"raw": 0, "final": 0, "generated": 0}
            continue
        needed = max(0, target_per_class - raw_count)
        if needed > 0:
            src_paths = [os.path.join(folder, f) for f in raw_images]
            for i in range(needed):
                src = src_paths[i % raw_count]
                img = load_image(src)
                aug = AUG_PIPELINE(image=img)["image"]
                cv2.imwrite(os.path.join(folder, f"aug_{i:05d}.jpg"), aug)
        aug_record[cls] = {"raw": raw_count, "final": len(list_images(folder)), "generated": needed}

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
    dirichlet_record = partition_dirichlet_train(pool_train, run3_dir, DIRICHLET_ALPHA, log, rng)

    aug_log = {}
    for c in range(1, NUM_CLIENTS + 1):
        ck = f"Client_{c}"
        aug_log[ck] = {}
        augment_client_train(os.path.join(run3_dir, ck), TARGET_PER_CLASS, aug_log[ck])

    master_log[fold_name] = {
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "dirichlet_raw_assignment": dirichlet_record,
        "raw_partition_log": log,
        "augmentation_log": aug_log,
    }

with open("dirichlet_label_skew_distribution_log.json", "w") as f:
    json.dump(master_log, f, indent=2)

# ── Sanity check ─────────────────────────────────────────────────────────
print("\nSanity check -- verifying Valid/Test parity with Run1, scanning for empty Train folders ...")
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
                if split == "Train" and n_run3 == 0:
                    issues.append(f"  [EMPTY TRAIN] {fold}/{NEW_RUN_NAME}/Client_{c}/{split}/{cls} (check if expected under this alpha)")
if issues:
    for i in issues: print(i)
else:
    print("  All folders populated; Valid/Test exactly match FL_Run1_Uniform.")

print("\nDirichlet label-skew (FL_Run3) dataset generation complete.")