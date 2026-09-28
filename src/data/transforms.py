import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 512):
    """
    Softened augmentation pipeline for the training set.

    Key changes from previous pipeline:
      - Removed: GridDistortion, OpticalDistortion, CoarseDropout
        (these destroy fine lesion geometry — microaneurysms, hemorrhages)
      - Reduced: ColorJitter hue 0.1→0.02, RandomResizedCrop scale (0.8,1.0)→(0.9,1.0)
      - Kept: RandomRotate90, flips, CLAHE, Sharpen, RandomBrightnessContrast, Affine

    NOTE: Uses `size=(h, w)` API for newer albumentations versions (>=1.4).
    """
    return A.Compose([
        # RandomResizedCrop — softened scale range to preserve peripheral lesions
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.9, 1.0),        # crop between 90%-100% (was 80%-100%)
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_CUBIC,
            p=1.0
        ),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Reduced hue from 0.1 to 0.02 — color is diagnostic in fundus images
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        # Removed: GridDistortion, OpticalDistortion, CoarseDropout
        # Affine — mild geometric augmentation (kept, was already reasonable)
        A.Affine(translate_percent=0.1, scale=0.9, rotate=45, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_test_transforms(img_size: int = 512):
    """
    Validation / Test / External transform pipeline.
    No augmentation — only resize, normalisation, and tensor conversion.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])



def get_tta_transforms(img_size: int = 512, scale: float = 1.0):
    """
    Transform pipeline for a single TTA view at a given scale.
    The scale parameter controls the multi-scale crop size before final resize.
    """
    crop_size = int(img_size * scale)
    return A.Compose([
        A.Resize(height=crop_size, width=crop_size),
        A.Resize(height=img_size, width=img_size),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

