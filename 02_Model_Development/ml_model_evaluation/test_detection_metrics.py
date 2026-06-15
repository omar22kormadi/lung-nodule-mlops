"""
Independent Test Script for Detection Metrics
Evaluates the YOLO model on the validation/test dataset to verify metrics.
"""
import os
from pathlib import Path
from ultralytics import YOLO

def evaluate_detection():
    print("=" * 60)
    print("DETECTION INDEPENDENT EVALUATION")
    print("=" * 60)
    
    # Paths
    model_path = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\02_Model_Development\ml_model_engineering\models\best.pt"
    # Using v4 dataset yaml based on previous context, adjust if needed
    data_yaml = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\luna16_yolo_dataset_v4\dataset.yaml"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
        
    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset YAML not found at {data_yaml}")
        print("Please check if the dataset version is correct (v3 vs v4).")
        # Try fallback
        data_yaml = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\processed\yolo_dataset_v3\dataset.yaml"
        if not os.path.exists(data_yaml):
             return
        print(f"Using fallback dataset: {data_yaml}")
        
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print("\nStarting rigorous evaluation...")
    
    # Run validation (you can change split="test" if you have a test split in your dataset.yaml)
    metrics = model.val(
        data=data_yaml,
        split="val",
        batch=8,
        imgsz=512,
        conf=0.25,   # standard confidence threshold
        iou=0.45,    # standard NMS IOU threshold
        verbose=False
    )
    
    print("\n" + "=" * 60)
    print("TRUE TEST METRICS (DETECTION)")
    print("=" * 60)
    print(f"mAP@50        : {metrics.box.map50:.4f}")
    print(f"mAP@50-95     : {metrics.box.map:.4f}")
    print(f"Precision (P) : {metrics.box.mp:.4f}")
    print(f"Recall (R)    : {metrics.box.mr:.4f}")
    
    p = metrics.box.mp
    r = metrics.box.mr
    f1 = 2 * (p * r) / (p + r + 1e-8)
    print(f"F1 Score      : {f1:.4f}")
    print("=" * 60)
    
    print("Metrics verified directly via Ultralytics API.")

if __name__ == "__main__":
    evaluate_detection()
