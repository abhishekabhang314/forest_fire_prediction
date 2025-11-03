# forest_fire_detection/src/data_preprocessing.py

import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine
import cv2
from tqdm import tqdm


def convert_png_pgw_to_geotiff(img_path, pgw_path, out_path):
    """
    Converts PNG + PGW world file to GeoTIFF with WGS84 CRS.
    """
    # Read world file
    with open(pgw_path, "r") as f:
        vals = [float(x.strip()) for x in f.readlines()]
    A, D, B, E, C, F = vals  # affine coefficients
    transform = Affine(A, B, C, D, E, F)

    # Read image
    img = np.array(Image.open(img_path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # Write GeoTIFF
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=img.shape[0],
        width=img.shape[1],
        count=3,
        dtype=img.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(img[:, :, i], i + 1)

    return out_path


def extract_fire_mask(image_path, save_path):
    """
    Detects orange fire overlay and saves a binary mask.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define orange color range (tweak as needed)
    lower = np.array([150, 50, 0])
    upper = np.array([255, 180, 100])
    mask = cv2.inRange(img, lower, upper)

    cv2.imwrite(save_path, mask)
    return save_path


def process_region(region_dir):
    """
    Converts all PNG+PGW pairs and extracts masks.
    """
    raw_dir = os.path.join(region_dir, "raw")
    processed_dir = os.path.join(region_dir, "processed")

    os.makedirs(os.path.join(processed_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "masks"), exist_ok=True)

    png_files = [f for f in os.listdir(os.path.join(raw_dir, "images")) if f.endswith(".png")]

    for fname in tqdm(png_files, desc=f"Processing {region_dir}"):
        img_path = os.path.join(raw_dir, "images", fname)
        pgw_path = os.path.join(raw_dir, "pgw", fname.replace(".png", ".pgw"))
        out_tif = os.path.join(processed_dir, "images", fname.replace(".png", ".tif"))
        mask_path = os.path.join(processed_dir, "masks", fname)

        convert_png_pgw_to_geotiff(img_path, pgw_path, out_tif)
        extract_fire_mask(img_path, mask_path)

    print(f"✅ Processing complete for {region_dir}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    ALBERTA_DIR = os.path.join(BASE_DIR, "data", "alberta")
    process_region(ALBERTA_DIR)
