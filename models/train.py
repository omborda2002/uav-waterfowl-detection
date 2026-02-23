"""
YOLOv8 Training Script for UAV Waterfowl Detection
Includes: TensorBoard logging, early stopping, model checkpointing, LR scheduling
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
from datetime import datetime
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data.config import *


class WaterfowlTrainer:
    """Complete training pipeline for waterfowl detection"""
    
    def __init__(self, model_name=YOLO_MODEL, experiment_name=None):
        """
        Initialize trainer
        
        Args:
            model_name: YOLO model variant (yolov8n.pt, yolov8s.pt, etc.)
            experiment_name: Name for this training run (auto-generated if None)
        """
        self.model_name = model_name
        
        # Generate experiment name with timestamp
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"waterfowl_{model_name.replace('.pt', '')}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Setup paths
        self.experiment_dir = WEIGHTS_PATH / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_yaml = get_yolo_data_yaml_path()
        
        # Initialize model
        self.model = None
        self.results = None
        
        print(f"\n{'='*80}")
        print(f"WATERFOWL DETECTION TRAINING")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Model: {model_name}")
        print(f"Output directory: {self.experiment_dir}")
        print(f"{'='*80}\n")
    
    def verify_setup(self):
        """Verify all prerequisites are met"""
        print("\n" + "="*80)
        print("VERIFYING SETUP")
        print("="*80)
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"\n✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ WARNING: CUDA not available. Training will be VERY slow on CPU!")
            print("  Consider using a machine with GPU for faster training.")
        
        # Check data.yaml exists
        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml not found at {self.data_yaml}\n"
                "Please run data preparation (Step 1) first!"
            )
        print(f"\n✓ Data config found: {self.data_yaml}")
        
        # Load and verify data.yaml
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"\nDataset configuration:")
        print(f"  Path: {data_config['path']}")
        print(f"  Classes: {data_config['names']}")
        print(f"  Num classes: {data_config['nc']}")
        
        # Check if dataset directories exist
        dataset_path = Path(data_config['path'])
        for split in ['train', 'val', 'test']:
            img_dir = dataset_path / 'images' / split
            lbl_dir = dataset_path / 'labels' / split
            
            if not img_dir.exists() or not lbl_dir.exists():
                raise FileNotFoundError(f"Dataset split '{split}' not found at {dataset_path}")
            
            num_images = len(list(img_dir.glob('*.tif')))
            num_labels = len(list(lbl_dir.glob('*.txt')))
            print(f"  {split.upper()}: {num_images} images, {num_labels} labels")
        
        print("\n✓ All prerequisites verified!")
        print("="*80)
    
    def load_model(self):
        """Load YOLOv8 model"""
        print(f"\nLoading model: {self.model_name}")
        
        self.model = YOLO(self.model_name)
        
        print(f"✓ Model loaded successfully")
        print(f"  Architecture: {self.model_name.replace('.pt', '').upper()}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
    
    def train(self, resume=False):
        """
        Train the model with all bells and whistles
        
        Args:
            resume: Resume training from last checkpoint
        """
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        # Prepare training arguments
        train_args = {
            # Data
            'data': str(self.data_yaml),
            
            # Training duration
            'epochs': EPOCHS,
            'batch': BATCH_SIZE,
            
            # Image settings
            'imgsz': YOLO_IMG_SIZE,
            
            # Optimization
            'optimizer': OPTIMIZER,
            'lr0': LEARNING_RATE,
            'lrf': 0.01,  # Final learning rate (lr0 * lrf)
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # Augmentation (thermal-specific)
            'hsv_h': AUGMENTATION['hsv_h'],
            'hsv_s': AUGMENTATION['hsv_s'],
            'hsv_v': AUGMENTATION['hsv_v'],
            'degrees': AUGMENTATION['degrees'],
            'translate': AUGMENTATION['translate'],
            'scale': AUGMENTATION['scale'],
            'shear': AUGMENTATION['shear'],
            'perspective': AUGMENTATION['perspective'],
            'flipud': AUGMENTATION['flipud'],
            'fliplr': AUGMENTATION['fliplr'],
            'mosaic': AUGMENTATION['mosaic'],
            'mixup': AUGMENTATION['mixup'],
            
            # Training settings
            'patience': PATIENCE,  # Early stopping patience
            'save': True,  # Save checkpoints
            'save_period': 10,  # Save every 10 epochs
            'cache': False,  # Don't cache images (can use lots of RAM)
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 8,  # Number of dataloader workers
            'project': str(WEIGHTS_PATH),  # Save location
            'name': self.experiment_name,
            'exist_ok': True,  # Overwrite existing
            'pretrained': True,  # Use pretrained weights
            'verbose': True,  # Verbose output
            
            # Validation
            'val': True,  # Validate during training
            'plots': True,  # Generate plots
            
            # Resume
            'resume': resume,
        }
        
        print("\nTraining configuration:")
        print(f"  Epochs: {EPOCHS}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Image size: {YOLO_IMG_SIZE}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Optimizer: {OPTIMIZER}")
        print(f"  Early stopping patience: {PATIENCE}")
        print(f"  Device: {train_args['device']}")
        print(f"  Workers: {train_args['workers']}")
        
        print("\nAugmentation settings:")
        for key, value in AUGMENTATION.items():
            print(f"  {key}: {value}")
        
        print("\n" + "-"*80)
        print("Training will start in 3 seconds...")
        print("Press Ctrl+C to stop training gracefully")
        print("-"*80 + "\n")
        
        import time
        time.sleep(3)
        
        # Train model
        try:
            self.results = self.model.train(**train_args)
            
            print("\n" + "="*80)
            print("✓ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            # Get best model path
            best_model_path = WEIGHTS_PATH / self.experiment_name / "weights" / "best.pt"
            last_model_path = WEIGHTS_PATH / self.experiment_name / "weights" / "last.pt"
            
            print(f"\nModel weights saved:")
            print(f"  Best: {best_model_path}")
            print(f"  Last: {last_model_path}")
            
            # Copy best model to main weights directory
            final_model_path = WEIGHTS_PATH / f"{self.experiment_name}_best.pt"
            shutil.copy2(best_model_path, final_model_path)
            print(f"  Final: {final_model_path}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("⚠ TRAINING INTERRUPTED BY USER")
            print("="*80)
            print("\nPartial results have been saved.")
            print("You can resume training by setting resume=True")
            return False
        
        except Exception as e:
            print("\n\n" + "="*80)
            print("❌ TRAINING FAILED")
            print("="*80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_training_summary(self):
        """Print training summary"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        results_dir = WEIGHTS_PATH / self.experiment_name
        
        # Check if results exist
        if not results_dir.exists():
            print("No training results found!")
            return
        
        # List generated files
        print("\nGenerated files:")
        
        weights_dir = results_dir / "weights"
        if weights_dir.exists():
            print(f"\n  Weights ({weights_dir}):")
            for weight_file in weights_dir.glob("*.pt"):
                size_mb = weight_file.stat().st_size / (1024 * 1024)
                print(f"    - {weight_file.name} ({size_mb:.2f} MB)")
        
        # Results files
        results_files = [
            "results.png",
            "results.csv",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
            "PR_curve.png",
        ]
        
        print(f"\n  Results ({results_dir}):")
        for filename in results_files:
            filepath = results_dir / filename
            if filepath.exists():
                print(f"    ✓ {filename}")
        
        # Try to read final metrics
        csv_path = results_dir / "results.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            print("\n" + "-"*80)
            print("Final Metrics (Last Epoch):")
            print("-"*80)
            
            last_row = df.iloc[-1]
            
            metrics_to_show = {
                'metrics/precision(B)': 'Precision',
                'metrics/recall(B)': 'Recall',
                'metrics/mAP50(B)': 'mAP@0.5',
                'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
            }
            
            for col, name in metrics_to_show.items():
                col_clean = col.strip()
                if col_clean in df.columns:
                    value = last_row[col_clean]
                    print(f"  {name:20s}: {value:.4f}")
        
        print("\n" + "="*80)
        print("✓ Training complete! Proceed to evaluation (Step 3)")
        print("="*80)


def main():
    """Main training function"""
    
    # Print header
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "YOLOV8 TRAINING PIPELINE" + " "*28 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Initialize trainer
    trainer = WaterfowlTrainer(model_name=YOLO_MODEL)
    
    # Verify setup
    trainer.verify_setup()
    
    # Load model
    trainer.load_model()
    
    # Train
    success = trainer.train(resume=False)
    
    if success:
        # Print summary
        trainer.get_training_summary()
        
        print("\n" + "🎉"*40)
        print("\n✓ TRAINING SUCCESSFUL!")
        print(f"\n📁 Results saved to: {trainer.experiment_dir}")
        print(f"📊 View training curves: {trainer.experiment_dir}/results.png")
        print(f"🎯 Best model: {WEIGHTS_PATH}/{trainer.experiment_name}_best.pt")
        print("\n⏭️  Next: Run evaluation script (Step 3)")
        print("\n" + "🎉"*40 + "\n")
        
        return 0
    else:
        print("\n⚠️  Training did not complete successfully")
        print("Check error messages above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())