"""
Quick fix: Convert grayscale thermal images to 3-channel RGB format
YOLOv8 expects 3 channels, but our thermal images are single-channel grayscale
"""

import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data.config import *


def convert_grayscale_to_rgb(image_path):
    """Convert grayscale image to 3-channel RGB by replicating the channel"""
    # Open image
    img = Image.open(image_path)
    
    # If already RGB, skip
    if img.mode == 'RGB':
        return False
    
    # Convert grayscale to RGB by replicating the channel
    if img.mode == 'L':  # Grayscale
        img_rgb = Image.merge('RGB', (img, img, img))
        img_rgb.save(image_path)
        return True
    
    return False


def convert_all_images():
    """Convert all images in train/val/test splits"""
    print("\n" + "="*80)
    print("CONVERTING GRAYSCALE IMAGES TO RGB FORMAT")
    print("="*80)
    print("\nYOLOv8 requires 3-channel RGB images.")
    print("Converting thermal grayscale images by replicating the single channel...")
    
    total_converted = 0
    
    for split in ['train', 'val', 'test']:
        img_dir = YOLO_IMAGES_PATH / split
        
        if not img_dir.exists():
            print(f"\n✗ Directory not found: {img_dir}")
            continue
        
        # Get all images
        images = list(img_dir.glob('*.tif'))
        
        print(f"\n{split.upper()}: Processing {len(images)} images...")
        
        converted_count = 0
        for img_path in tqdm(images, desc=f"  Converting {split}"):
            if convert_grayscale_to_rgb(img_path):
                converted_count += 1
        
        print(f"  ✓ Converted: {converted_count}/{len(images)} images")
        total_converted += converted_count
    
    print("\n" + "="*80)
    print(f"✓ CONVERSION COMPLETE!")
    print(f"  Total images converted: {total_converted}")
    print("="*80)
    print("\n✓ You can now run training!")


if __name__ == "__main__":
    convert_all_images()