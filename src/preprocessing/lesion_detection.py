"""
Classical Lesion Detection Module for Diabetic Retinopathy.

Detects three key lesion types using traditional computer vision:
  1. Microaneurysms — small dark dots (3-15px radius)
  2. Hemorrhages — larger dark regions (>20px area)
  3. Exudates — bright yellow-white lesions

These auto-generated pseudo-labels are used as supervision signal
for the Lesion-Aware Self-Supervised Pretraining (Phase 3 of plan.md).

Outputs per image:
  - Lesion type mask: 3 binary masks (H, W, 3) for [microaneurysm, hemorrhage, exudate]
  - Lesion count: int[3] — count of each lesion type
  - Lesion presence: float[3] — binary multi-label (0 or 1)
  - Lesion location heatmap: float (H, W) — Gaussian-blurred density map

Author: RetiNA-Net Project
"""

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


class LesionDetector:
    """
    Classical lesion detector using OpenCV-based image processing.

    All detection methods operate on the green channel of the fundus image,
    which provides the highest contrast for vascular lesions.
    """

    def __init__(self, mask_size=128):
        """
        Args:
            mask_size: Output mask size (mask_size x mask_size).
                      All masks are resized to this resolution.
        """
        self.mask_size = mask_size

    def _preprocess(self, img):
        """
        Preprocess fundus image for lesion detection.
        Returns green channel and CLAHE-enhanced green channel.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            green: Green channel (H, W)
            green_clahe: CLAHE-enhanced green channel (H, W)
        """
        # Extract green channel (best contrast for lesions)
        green = img[:, :, 1]

        # Apply CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green_clahe = clahe.apply(green)

        return green, green_clahe

    def detect_microaneurysms(self, img):
        """
        Detect microaneurysms using morphological top-hat + Hough circles.

        Microaneurysms appear as small dark circular spots (3-15px radius)
        in the green channel.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            mask: Binary mask (H, W) — 1 where microaneurysms detected
            count: Number of microaneurysms detected
        """
        green, green_clahe = self._preprocess(img)
        h, w = green.shape

        # Morphological top-hat to isolate small dark spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(green_clahe, cv2.MORPH_TOPHAT, kernel)

        # Threshold to keep only significant dark spots
        _, binary = cv2.threshold(tophat, 12, 255, cv2.THRESH_BINARY)

        # Remove noise with morphological opening
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # Hough circle detection for circular structures
        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=50,
            param2=8,
            minRadius=3,
            maxRadius=15
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        count = 0

        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            count = len(circles)
            for (cx, cy, r) in circles:
                # Draw filled circle at detected microaneurysm location
                cv2.circle(mask, (cx, cy), int(r), 1, thickness=-1)

        # Also add small connected components from binary mask
        # (catches microaneurysms missed by Hough)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 5 <= area <= 100:  # Microaneurysm-sized regions
                mask[labels == i] = 1
                count += 1

        # Cap count to avoid extreme outliers
        count = min(count, 100)

        return mask, count

    def detect_hemorrhages(self, img):
        """
        Detect hemorrhages using adaptive thresholding + connected components.

        Hemorrhages appear as larger dark regions (>20px area) that are
        not circular (unlike microaneurysms).

        Args:
            img: BGR image (H, W, 3)

        Returns:
            mask: Binary mask (H, W) — 1 where hemorrhages detected
            count: Number of hemorrhage regions detected
        """
        green, green_clahe = self._preprocess(img)
        h, w = green.shape

        # Adaptive thresholding to find dark regions
        binary = cv2.adaptiveThreshold(
            green_clahe, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=51,
            C=5
        )

        # Remove microaneurysm-sized regions (keep only larger dark areas)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Connected components with area filter
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        mask = np.zeros((h, w), dtype=np.uint8)
        count = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 20:  # Hemorrhage-sized regions
                mask[labels == i] = 1
                count += 1

        # Cap count
        count = min(count, 50)

        return mask, count

    def detect_exudates(self, img):
        """
        Detect exudates using LAB color space thresholding.

        Exudates appear as bright yellow-white lesions. We threshold on
        high L (lightness) and B (yellow) channels in LAB color space,
        then filter by area and exclude the optic disc region.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            mask: Binary mask (H, W) — 1 where exudates detected
            count: Number of exudate regions detected
        """
        h, w = img.shape[:2]

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        # Threshold: high lightness AND high yellow (b channel)
        _, l_thresh = cv2.threshold(l_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, b_thresh = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine: exudates = bright AND yellow
        binary = cv2.bitwise_and(l_thresh, b_thresh)

        # Morphological closing to merge nearby exudates
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Remove optic disc: find brightest circular region and exclude it
        # The optic disc is typically the brightest region near the edge
        # Use morphological opening with large kernel to separate exudates from disc
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_large)

        # Connected components with area filter
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        mask = np.zeros((h, w), dtype=np.uint8)
        count = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 10 <= area <= 5000:  # Exudate-sized regions
                mask[labels == i] = 1
                count += 1

        # Cap count
        count = min(count, 50)

        return mask, count

    def detect_all(self, img):
        """
        Run all three lesion detectors on an image.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            dict with:
              - 'mask': (mask_size, mask_size, 3) — binary masks for [micro, hem, exu]
              - 'count': np.array([micro_count, hem_count, exu_count])
              - 'presence': np.array([micro_present, hem_present, exu_present]) — binary
              - 'heatmap': (mask_size, mask_size) — Gaussian-blurred density map
        """
        h, w = img.shape[:2]

        # Detect each lesion type
        ma_mask, ma_count = self.detect_microaneurysms(img)
        hem_mask, hem_count = self.detect_hemorrhages(img)
        exu_mask, exu_count = self.detect_exudates(img)

        # Resize masks to standard size
        size = self.mask_size
        ma_mask_resized = cv2.resize(ma_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        hem_mask_resized = cv2.resize(hem_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        exu_mask_resized = cv2.resize(exu_mask, (size, size), interpolation=cv2.INTER_NEAREST)

        # Stack into (mask_size, mask_size, 3)
        mask = np.stack([ma_mask_resized, hem_mask_resized, exu_mask_resized], axis=-1).astype(np.float32)

        # Counts
        counts = np.array([ma_count, hem_count, exu_count], dtype=np.float32)

        # Presence (binary)
        presence = (counts > 0).astype(np.float32)

        # Heatmap: Gaussian-blurred sum of all lesion masks
        combined = (ma_mask + hem_mask + exu_mask).astype(np.float32)
        if combined.max() > 0:
            heatmap = cv2.GaussianBlur(combined, (31, 31), 0)
            # Normalize to [0, 1]
            max_val = heatmap.max()
            if max_val > 0:
                heatmap = heatmap / max_val
        else:
            heatmap = np.zeros((h, w), dtype=np.float32)

        heatmap = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_LINEAR)

        return {
            'mask': mask,           # (mask_size, mask_size, 3)
            'count': counts,        # (3,)
            'presence': presence,   # (3,)
            'heatmap': heatmap      # (mask_size, mask_size)
        }


def detect_lesions_batch(image_dirs, output_dir, mask_size=128, extensions=('.jpeg', '.jpg', '.png')):
    """
    Run lesion detection on all images in the given directories.
    Save results as .npz files for use during SSL pretraining.

    Each .npz file contains:
      - mask: (mask_size, mask_size, 3) float32
      - count: (3,) float32
      - presence: (3,) float32
      - heatmap: (mask_size, mask_size) float32

    Args:
        image_dirs: List of directories containing fundus images
        output_dir: Directory to save .npz lesion label files
        mask_size: Output mask resolution
        extensions: Tuple of valid image extensions
    """
    os.makedirs(output_dir, exist_ok=True)
    detector = LesionDetector(mask_size=mask_size)

    # Collect all image paths
    all_images = []
    for img_dir in image_dirs:
        for ext in extensions:
            all_images.extend(glob(os.path.join(img_dir, f'*{ext}')))
            all_images.extend(glob(os.path.join(img_dir, f'*{ext.upper()}')))

    print(f"Found {len(all_images)} images. Running lesion detection...")

    for img_path in tqdm(all_images, desc="Detecting lesions"):
        # Output filename: same name as image but .npz
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, basename + '.npz')

        # Skip if already processed
        if os.path.exists(out_path):
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue

        # Detect lesions
        try:
            results = detector.detect_all(img)
            np.savez_compressed(out_path,
                                mask=results['mask'],
                                count=results['count'],
                                presence=results['presence'],
                                heatmap=results['heatmap'])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Lesion detection complete. Results saved to {output_dir}/")


if __name__ == '__main__':
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Classical Lesion Detection")
    parser.add_argument('--image_dirs', nargs='+', required=True, help='Directories containing fundus images')
    parser.add_argument('--output_dir', required=True, help='Output directory for .npz lesion labels')
    parser.add_argument('--mask_size', type=int, default=128, help='Output mask size')
    args = parser.parse_args()

    detect_lesions_batch(args.image_dirs, args.output_dir, args.mask_size)
