"""
Data Preparation Script for UAV Waterfowl Detection
Converts CSV annotations to YOLO format and creates train/val/test splits
"""

import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import yaml
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *


class DatasetPreparer:
    """Handles dataset preparation and conversion to YOLO format"""
    
    def __init__(self):
        self.df = None
        self.image_files = []
        self.splits = {}
        
    def load_annotations(self):
        """Load and parse the CSV annotation file"""
        print("\n" + "="*80)
        print("LOADING ANNOTATIONS")
        print("="*80)
        
        if not ANNOTATIONS_PATH.exists():
            raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_PATH}")
        
        # Load CSV
        self.df = pd.read_csv(ANNOTATIONS_PATH)
        
        print(f"✓ Loaded {len(self.df)} annotations from {ANNOTATIONS_PATH.name}")
        print(f"  Columns: {list(self.df.columns)}")
        print(f"  Unique images: {self.df['imageFilename'].nunique()}")
        
        # Basic statistics
        birds_per_image = self.df.groupby('imageFilename').size()
        print(f"\n  Birds per image:")
        print(f"    Mean: {birds_per_image.mean():.2f}")
        print(f"    Median: {birds_per_image.median():.2f}")
        print(f"    Min: {birds_per_image.min()}")
        print(f"    Max: {birds_per_image.max()}")
        
        return self.df
    
    def collect_images(self):
        """Collect all positive and negative image paths"""
        print("\n" + "="*80)
        print("COLLECTING IMAGES")
        print("="*80)
        
        # Collect positive images (with annotations)
        positive_images = []
        if POSITIVE_IMAGES_PATH.exists():
            positive_files = list(POSITIVE_IMAGES_PATH.glob("*.tif"))
            positive_images = [{'path': f, 'has_objects': True} for f in positive_files]
            print(f"✓ Found {len(positive_images)} positive images")
        
        # Collect negative images (no birds)
        negative_images = []
        if NEGATIVE_IMAGES_PATH.exists():
            negative_files = list(NEGATIVE_IMAGES_PATH.glob("*.tif"))
            negative_images = [{'path': f, 'has_objects': False} for f in negative_files]
            print(f"✓ Found {len(negative_images)} negative images")
        
        # Combine all images
        self.image_files = positive_images + negative_images
        
        print(f"\n✓ Total images: {len(self.image_files)}")
        print(f"  Positive: {len(positive_images)} ({len(positive_images)/len(self.image_files)*100:.1f}%)")
        print(f"  Negative: {len(negative_images)} ({len(negative_images)/len(self.image_files)*100:.1f}%)")
        
        return self.image_files
    
    def convert_bbox_to_yolo(self, x, y, width, height, img_width, img_height):
        """
        Convert bounding box from (x, y, width, height) to YOLO format
        YOLO format: (x_center, y_center, width, height) normalized to [0, 1]
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Box dimensions
            img_width, img_height: Image dimensions
            
        Returns:
            Tuple of (x_center_norm, y_center_norm, width_norm, height_norm)
        """
        # Calculate center coordinates
        x_center = x + width / 2
        y_center = y + height / 2
        
        # Normalize to [0, 1]
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Clip to [0, 1] range
        x_center_norm = np.clip(x_center_norm, 0, 1)
        y_center_norm = np.clip(y_center_norm, 0, 1)
        width_norm = np.clip(width_norm, 0, 1)
        height_norm = np.clip(height_norm, 0, 1)
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    def create_yolo_annotations(self, image_info, split):
        """
        Create YOLO format annotation file for an image
        
        Args:
            image_info: Dictionary with image path and metadata
            split: 'train', 'val', or 'test'
        """
        image_path = image_info['path']
        image_name = image_path.name
        has_objects = image_info['has_objects']
        
        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Create label file path
        label_filename = image_path.stem + '.txt'
        label_path = YOLO_LABELS_PATH / split / label_filename
        
        # If image has no objects (negative sample), create empty label file
        if not has_objects:
            label_path.write_text('')
            return label_path
        
        # Get annotations for this image
        image_annotations = self.df[self.df['imageFilename'] == image_name]
        
        if len(image_annotations) == 0:
            # No annotations found, create empty file
            label_path.write_text('')
            return label_path
        
        # Convert annotations to YOLO format
        yolo_annotations = []
        for _, row in image_annotations.iterrows():
            x = row['x(column)']
            y = row['y(row)']
            width = row['width']
            height = row['height']
            
            # Convert to YOLO format
            x_center, y_center, w_norm, h_norm = self.convert_bbox_to_yolo(
                x, y, width, height, img_width, img_height
            )
            
            # YOLO format: class_id x_center y_center width height
            # class_id = 0 for waterfowl (single class)
            yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_annotations.append(yolo_line)
        
        # Write to file
        label_path.write_text('\n'.join(yolo_annotations))
        
        return label_path
    
    def create_splits(self):
        """Create train/val/test splits"""
        print("\n" + "="*80)
        print("CREATING DATASET SPLITS")
        print("="*80)
        
        # Separate positive and negative images for stratified splitting
        positive_imgs = [img for img in self.image_files if img['has_objects']]
        negative_imgs = [img for img in self.image_files if not img['has_objects']]
        
        print(f"\nSplitting positive images ({len(positive_imgs)})...")
        # Split positive images
        pos_train, pos_temp = train_test_split(
            positive_imgs, 
            test_size=(VAL_SPLIT + TEST_SPLIT),
            random_state=RANDOM_SEED
        )
        pos_val, pos_test = train_test_split(
            pos_temp,
            test_size=TEST_SPLIT/(VAL_SPLIT + TEST_SPLIT),
            random_state=RANDOM_SEED
        )
        
        print(f"Splitting negative images ({len(negative_imgs)})...")
        # Split negative images
        neg_train, neg_temp = train_test_split(
            negative_imgs,
            test_size=(VAL_SPLIT + TEST_SPLIT),
            random_state=RANDOM_SEED
        )
        neg_val, neg_test = train_test_split(
            neg_temp,
            test_size=TEST_SPLIT/(VAL_SPLIT + TEST_SPLIT),
            random_state=RANDOM_SEED
        )
        
        # Combine positive and negative splits
        self.splits = {
            'train': pos_train + neg_train,
            'val': pos_val + neg_val,
            'test': pos_test + neg_test
        }
        
        # Print split statistics
        print("\n" + "-"*80)
        print("SPLIT STATISTICS")
        print("-"*80)
        for split_name, split_images in self.splits.items():
            pos_count = sum(1 for img in split_images if img['has_objects'])
            neg_count = sum(1 for img in split_images if not img['has_objects'])
            total = len(split_images)
            
            print(f"\n{split_name.upper()}:")
            print(f"  Total: {total} images ({total/len(self.image_files)*100:.1f}%)")
            print(f"  Positive: {pos_count} ({pos_count/total*100:.1f}%)")
            print(f"  Negative: {neg_count} ({neg_count/total*100:.1f}%)")
        
        return self.splits
    
    def copy_and_convert(self):
        """Copy images and create YOLO annotations for all splits"""
        print("\n" + "="*80)
        print("CONVERTING TO YOLO FORMAT")
        print("="*80)
        print("\nNote: Converting grayscale thermal images to 3-channel RGB")
        print("(YOLOv8 requires 3-channel input)")
        
        for split_name, split_images in self.splits.items():
            print(f"\nProcessing {split_name} split...")
            
            # Create progress bar
            for image_info in tqdm(split_images, desc=f"  Converting {split_name}"):
                image_path = image_info['path']
                dest_image_path = YOLO_IMAGES_PATH / split_name / image_path.name
                
                # Load image and convert to RGB if grayscale
                img = Image.open(image_path)
                if img.mode == 'L':  # Grayscale
                    # Convert to RGB by replicating the single channel
                    img = Image.merge('RGB', (img, img, img))
                img.save(dest_image_path)
                
                # Create YOLO annotation
                self.create_yolo_annotations(image_info, split_name)
            
            print(f"  ✓ Completed {split_name}: {len(split_images)} images")
    
    def create_data_yaml(self):
        """Create YOLO data.yaml configuration file"""
        print("\n" + "="*80)
        print("CREATING YOLO CONFIGURATION")
        print("="*80)
        
        # YOLO data.yaml structure
        data_yaml = {
            'path': str(YOLO_DATASET_PATH.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': NUM_CLASSES,
            'names': CLASS_NAMES
        }
        
        yaml_path = get_yolo_data_yaml_path()
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created YOLO configuration file: {yaml_path}")
        print("\nConfiguration:")
        print(yaml.dump(data_yaml, default_flow_style=False, sort_keys=False))
        
        return yaml_path
    
    def verify_dataset(self):
        """Verify the prepared dataset"""
        print("\n" + "="*80)
        print("VERIFYING DATASET")
        print("="*80)
        
        for split in ['train', 'val', 'test']:
            img_dir = YOLO_IMAGES_PATH / split
            label_dir = YOLO_LABELS_PATH / split
            
            num_images = len(list(img_dir.glob("*.tif")))
            num_labels = len(list(label_dir.glob("*.txt")))
            
            print(f"\n{split.upper()}:")
            print(f"  Images: {num_images}")
            print(f"  Labels: {num_labels}")
            
            if num_images != num_labels:
                print(f"  ⚠ Warning: Mismatch between images and labels!")
            else:
                print(f"  ✓ Match!")
            
            # Check a sample annotation
            sample_label = list(label_dir.glob("*.txt"))[0]
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                print(f"  Sample annotation ({sample_label.name}): {len(lines)} objects")
                if len(lines) > 0:
                    print(f"    First line: {lines[0].strip()}")
        
        print("\n" + "="*80)
        print("✓ DATASET PREPARATION COMPLETE!")
        print("="*80)
    
    def prepare_all(self):
        """Run complete data preparation pipeline"""
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*20 + "DATA PREPARATION PIPELINE" + " "*33 + "║")
        print("╚" + "="*78 + "╝")
        
        # Create directories
        create_directories()
        
        # Load annotations
        self.load_annotations()
        
        # Collect images
        self.collect_images()
        
        # Create splits
        self.create_splits()
        
        # Convert and copy
        self.copy_and_convert()
        
        # Create YOLO configuration
        self.create_data_yaml()
        
        # Verify
        self.verify_dataset()
        
        print("\n✓ All data preparation steps completed successfully!")
        print(f"\n📁 YOLO dataset location: {YOLO_DATASET_PATH}")
        print(f"📄 Configuration file: {get_yolo_data_yaml_path()}")


def main():
    """Main execution function"""
    try:
        preparer = DatasetPreparer()
        preparer.prepare_all()
        
    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)