"""
Model Evaluation Script for UAV Waterfowl Detection
Comprehensive evaluation with metrics, visualizations, and error analysis
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data.config import *


class WaterfowlEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path, experiment_name=None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model weights (.pt file)
            experiment_name: Name for evaluation outputs
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Setup experiment name
        if experiment_name is None:
            self.experiment_name = self.model_path.stem + "_eval"
        else:
            self.experiment_name = experiment_name
        
        # Create evaluation directory
        self.eval_dir = RESULTS_PATH / self.experiment_name
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz_dir = VISUALIZATIONS_PATH / self.experiment_name
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"\nLoading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✓ Model loaded successfully")
        
        # Data paths
        self.data_yaml = get_yolo_data_yaml_path()
        self.test_images_dir = YOLO_IMAGES_PATH / 'test'
        self.test_labels_dir = YOLO_LABELS_PATH / 'test'
        
        # Results storage
        self.predictions = []
        self.ground_truths = []
        self.metrics = {}
    
    def evaluate_on_test_set(self):
        """Run model evaluation on test set"""
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)
        
        # Run validation
        results = self.model.val(
            data=str(self.data_yaml),
            split='test',
            batch=BATCH_SIZE,
            imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=0 if torch.cuda.is_available() else 'cpu',
            plots=True,
            save_json=True,
            project=str(self.eval_dir),
            name='test_results',
            exist_ok=True,
        )
        
        # Extract metrics
        self.metrics = {
            'precision': float(results.box.p[0]) if hasattr(results.box, 'p') else 0.0,
            'recall': float(results.box.r[0]) if hasattr(results.box, 'r') else 0.0,
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'f1': float(results.box.f1[0]) if hasattr(results.box, 'f1') else 0.0,
        }
        
        print("\n" + "-"*80)
        print("TEST SET METRICS")
        print("-"*80)
        for metric_name, value in self.metrics.items():
            print(f"  {metric_name:15s}: {value:.4f}")
        print("-"*80)
        
        # Save metrics
        metrics_file = self.eval_dir / "test_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n✓ Metrics saved to: {metrics_file}")
        
        return results
    
    def analyze_predictions(self, conf_threshold=CONF_THRESHOLD):
        """Analyze predictions for error analysis"""
        print("\n" + "="*80)
        print("ANALYZING PREDICTIONS")
        print("="*80)
        
        test_images = list(self.test_images_dir.glob('*.tif'))
        
        print(f"\nProcessing {len(test_images)} test images...")
        
        # Storage for analysis
        true_positives = []
        false_positives = []
        false_negatives = []
        
        for img_path in tqdm(test_images, desc="Analyzing"):
            # Load ground truth
            label_path = self.test_labels_dir / (img_path.stem + '.txt')
            
            gt_boxes = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # class, x_center, y_center, width, height (normalized)
                            gt_boxes.append([float(x) for x in parts[1:]])
            
            # Get predictions
            results = self.model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=IOU_THRESHOLD,
                verbose=False,
            )[0]
            
            # Extract predicted boxes
            pred_boxes = []
            if len(results.boxes) > 0:
                boxes = results.boxes.xywhn.cpu().numpy()  # normalized xywh
                confs = results.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    pred_boxes.append({
                        'box': box.tolist(),
                        'conf': float(conf)
                    })
            
            # Match predictions with ground truth (simple IoU matching)
            matched_gt = set()
            matched_pred = set()
            
            for i, pred in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = self.calculate_iou(pred['box'], gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= IOU_THRESHOLD:
                    # True Positive
                    matched_pred.add(i)
                    matched_gt.add(best_gt_idx)
                    true_positives.append({
                        'image': img_path.name,
                        'confidence': pred['conf'],
                        'iou': best_iou
                    })
            
            # False Positives (predictions with no match)
            for i, pred in enumerate(pred_boxes):
                if i not in matched_pred:
                    false_positives.append({
                        'image': img_path.name,
                        'confidence': pred['conf'],
                        'box': pred['box']
                    })
            
            # False Negatives (ground truth with no match)
            for j, gt in enumerate(gt_boxes):
                if j not in matched_gt:
                    false_negatives.append({
                        'image': img_path.name,
                        'box': gt
                    })
        
        print(f"\n✓ Analysis complete!")
        print(f"  True Positives: {len(true_positives)}")
        print(f"  False Positives: {len(false_positives)}")
        print(f"  False Negatives: {len(false_negatives)}")
        
        # Save analysis
        analysis = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'summary': {
                'tp': len(true_positives),
                'fp': len(false_positives),
                'fn': len(false_negatives),
                'total_gt': len(true_positives) + len(false_negatives),
                'total_pred': len(true_positives) + len(false_positives),
            }
        }
        
        analysis_file = self.eval_dir / "error_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes in xywh format (normalized)"""
        # Convert to xyxy
        def xywh_to_xyxy(box):
            x, y, w, h = box
            return [x - w/2, y - h/2, x + w/2, y + h/2]
        
        box1_xyxy = xywh_to_xyxy(box1)
        box2_xyxy = xywh_to_xyxy(box2)
        
        # Calculate intersection
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box1_xyxy[3], box2_xyxy[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_examples(self, analysis, num_examples=3):
        """Visualize TP, FP, FN examples"""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        # Get examples for each category
        tp_images = set([x['image'] for x in analysis['true_positives'][:num_examples*10]])
        fp_images = set([x['image'] for x in analysis['false_positives'][:num_examples*10]])
        fn_images = set([x['image'] for x in analysis['false_negatives'][:num_examples*10]])
        
        # Visualize True Positives
        print(f"\nGenerating True Positive examples...")
        self._visualize_category(list(tp_images)[:num_examples], 'true_positives', 
                                'True Positives (Correct Detections)')
        
        # Visualize False Positives
        print(f"Generating False Positive examples...")
        self._visualize_category(list(fp_images)[:num_examples], 'false_positives',
                                'False Positives (Incorrect Detections)')
        
        # Visualize False Negatives
        print(f"Generating False Negative examples...")
        self._visualize_category(list(fn_images)[:num_examples], 'false_negatives',
                                'False Negatives (Missed Detections)')
        
        print("✓ Visualizations saved!")
    
    def _visualize_category(self, image_names, category, title):
        """Helper to visualize a category of examples"""
        if not image_names:
            print(f"  No examples for {category}")
            return
        
        fig, axes = plt.subplots(1, len(image_names), figsize=(5*len(image_names), 5))
        if len(image_names) == 1:
            axes = [axes]
        
        for idx, img_name in enumerate(image_names):
            img_path = self.test_images_dir / img_name
            label_path = self.test_labels_dir / (Path(img_name).stem + '.txt')
            
            # Read image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h, w = img.shape
            
            # Draw ground truth (blue)
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, x, y, bw, bh = map(float, parts)
                            x1 = int((x - bw/2) * w)
                            y1 = int((y - bh/2) * h)
                            x2 = int((x + bw/2) * w)
                            y2 = int((y + bh/2) * h)
                            cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw predictions (green)
            results = self.model.predict(source=str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            axes[idx].imshow(img_color)
            axes[idx].set_title(f"{Path(img_name).stem}", fontsize=8)
            axes[idx].axis('off')
        
        plt.suptitle(f"{title}\nBlue=Ground Truth, Green=Predictions", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.viz_dir / f"{category}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
    
    def create_summary_report(self, analysis):
        """Create a comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        report_path = self.eval_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WATERFOWL DETECTION - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model: {self.model_path.name}\n")
            f.write(f"Test set: {len(list(self.test_images_dir.glob('*.tif')))} images\n\n")
            
            f.write("-"*80 + "\n")
            f.write("METRICS\n")
            f.write("-"*80 + "\n")
            for metric, value in self.metrics.items():
                f.write(f"  {metric:15s}: {value:.4f}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("ERROR ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"  True Positives:  {analysis['summary']['tp']}\n")
            f.write(f"  False Positives: {analysis['summary']['fp']}\n")
            f.write(f"  False Negatives: {analysis['summary']['fn']}\n")
            f.write(f"  Total GT boxes:  {analysis['summary']['total_gt']}\n")
            f.write(f"  Total Pred boxes: {analysis['summary']['total_pred']}\n")
            
            # Calculate additional metrics
            tp = analysis['summary']['tp']
            fp = analysis['summary']['fp']
            fn = analysis['summary']['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f.write(f"\n  Precision (TP/TP+FP): {precision:.4f}\n")
            f.write(f"  Recall (TP/TP+FN):    {recall:.4f}\n")
            f.write(f"  F1 Score:             {f1:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("="*80 + "\n\n")
            
            # Interpretation
            if self.metrics['mAP50'] >= 0.80:
                f.write("✓ Excellent detection performance (mAP@0.5 >= 0.80)\n")
            elif self.metrics['mAP50'] >= 0.70:
                f.write("✓ Good detection performance (mAP@0.5 >= 0.70)\n")
            elif self.metrics['mAP50'] >= 0.60:
                f.write("○ Moderate detection performance (mAP@0.5 >= 0.60)\n")
            else:
                f.write("⚠ Performance could be improved (mAP@0.5 < 0.60)\n")
            
            f.write("\nStrengths and Weaknesses:\n")
            if precision > recall:
                f.write("  + High precision: Model makes few false positive errors\n")
                f.write("  - Lower recall: Model misses some waterfowl (false negatives)\n")
            elif recall > precision:
                f.write("  + High recall: Model detects most waterfowl\n")
                f.write("  - Lower precision: Model makes some false positive errors\n")
            else:
                f.write("  ○ Balanced precision and recall\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Report saved to: {report_path}")
        
        # Print report to console
        with open(report_path, 'r') as f:
            print("\n" + f.read())


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained waterfowl detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--name', type=str, default=None, help='Evaluation experiment name')
    
    args = parser.parse_args()
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*23 + "MODEL EVALUATION PIPELINE" + " "*29 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Initialize evaluator
    evaluator = WaterfowlEvaluator(args.model, args.name)
    
    # Run evaluation
    evaluator.evaluate_on_test_set()
    
    # Analyze predictions
    analysis = evaluator.analyze_predictions()
    
    # Create visualizations
    evaluator.visualize_examples(analysis, num_examples=3)
    
    # Generate report
    evaluator.create_summary_report(analysis)
    
    print("\n" + "🎉"*40)
    print("\n✓ EVALUATION COMPLETE!")
    print(f"\n📁 Results saved to: {evaluator.eval_dir}")
    print(f"📊 Visualizations: {evaluator.viz_dir}")
    print("\n" + "🎉"*40 + "\n")


if __name__ == "__main__":
    main()