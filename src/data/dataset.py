import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Suppress OpenCV warnings (imread errors flood the log)
# Use environment variable for compatibility with all OpenCV versions (setLogLevel requires 4.8+)
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'


class DRDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy grading.

    Automated OpenCV preprocessing pipeline (executed in __getitem__):
      1. Fundus boundary masking (crop black padding)
      2. CLAHE on L-channel of LAB (enhances microaneurysm contrast)
      3. Ben Graham's luminosity normalisation
      4. Albumentations transform (handles resize + augment + normalise)

    NOTE: The final resize to img_size is delegated to the Albumentations
    transform pipeline — we do NOT hardcode a cv2.resize here, avoiding
    the double-resize quality degradation present in the original code.
    """

    def __init__(self, image_dir: str, labels_df: pd.DataFrame, transform=None, img_size: int = 384):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            labels_df (pd.DataFrame): DataFrame with 'id_code' and 'diagnosis' columns.
            transform (albumentations.Compose, optional): Transform pipeline.
            img_size (int): Target image size (for reference; actual resize is in transform).
        """
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.labels_df)

    def _load_image(self, img_name):
        """
        Load image trying multiple extensions (.png, .jpg, .jpeg).
        Returns None if image cannot be found in any format.
        Uses os.path.exists() to avoid OpenCV warnings on missing files.
        """
        # If extension already provided, try as-is first
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(self.image_dir, img_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    return img

        # Try each extension
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(self.image_dir, img_name + ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    return img

        return None

    def _crop_fundus(self, img):
        """Crop the circular fundus region from the black background."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Threshold to isolate the fundus contour
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Add a small margin to avoid cutting off edge lesions
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            img = img[y:y + h, x:x + w]

        return img

    def _apply_clahe(self, img):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
        L-channel of the LAB color space. This significantly enhances the
        visibility of microaneurysms, hemorrhages, and exudates — the key
        lesions for DR grading.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE with clip limit and tile grid size tuned for retinal images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return img_clahe

    def _apply_ben_graham(self, img):
        """
        Ben Graham's luminosity standardisation:
        subtracts a heavily blurred version of the image to normalise
        illumination differences across different fundus cameras.
        """
        return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)

    def __getitem__(self, idx):
        # 1. Image path resolution
        row = self.labels_df.iloc[idx]
        img_name = str(row['id_code'])

        # 2. Load image (tries .png, .jpg, .jpeg)
        img = self._load_image(img_name)
        if img is None:
            # Graceful fallback: return a black placeholder image so training doesn't crash
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            # 3. Automated preprocessing pipeline
            # Step 1: Fundus boundary masking (crop black padding)
            img = self._crop_fundus(img)

            # Step 2: CLAHE enhancement (enhances lesion contrast)
            img = self._apply_clahe(img)

            # Step 3: Ben Graham's luminosity normalisation
            img = self._apply_ben_graham(img)

        # 4. Convert to RGB (Albumentations expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 5. Apply Albumentations transform (handles resize + augment + normalise)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        label = int(row['diagnosis'])
        return img, label
