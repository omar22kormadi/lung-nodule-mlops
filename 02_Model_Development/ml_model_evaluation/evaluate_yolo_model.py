import os
from pathlib import Path
from ultralytics import YOLO

def evaluate_model():
    # Paths
    model_path = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\02_Model_Development\ml_model_engineering\best.pt"
    data_yaml = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\luna16_yolo_dataset\dataset.yaml"
    output_dir = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\02_Model_Development\ml_model_evaluation\yolo_eval_output"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
        
    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset YAML not found at {data_yaml}")
        print("Please update the data_yaml path in this script if it's located elsewhere.")
        return
        
    print(f"Loading model from: {model_path}")
    print("="*50)
    
    # 1. Load the model
    # Ultralytics YOLO will automatically detect the model architecture (v8m, v8x, etc.) from the .pt file
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Print model info
    print("\nModel Info:")
    print(f"Task: {model.task}")
    print(model.info())
    print("="*50)
    
    # 2. Evaluate the model
    print("\nStarting evaluation on validation set...")
    try:
        # Run validation
        metrics = model.val(
            data=data_yaml,
            project=output_dir,
            name="evaluation_results",
            exist_ok=True,
            split="val", # You can change this to "test" if you have a test split in your dataset.yaml
            plots=True,  # Generate confusion matrix, PR curves, etc.
            batch=16,    # Adjust based on your GPU memory
            imgsz=512,   # Use the same image size used during training
            conf=0.25,   # Confidence threshold for evaluation
            iou=0.45     # NMS IOU threshold
        )
        
        # 3. Display Results
        print("\n" + "="*50)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("="*50)
        print(f"mAP@50        : {metrics.box.map50:.4f}")
        print(f"mAP@50-95     : {metrics.box.map:.4f}")
        print(f"Precision (P) : {metrics.box.mp:.4f}")
        print(f"Recall (R)    : {metrics.box.mr:.4f}")
        
        # Calculate F1 Score
        p = metrics.box.mp
        r = metrics.box.mr
        f1 = 2 * (p * r) / (p + r + 1e-8)
        print(f"F1 Score      : {f1:.4f}")
        print("="*50)
        
        print(f"\n✅ Evaluation complete! Detailed plots and metrics saved to:\n{os.path.join(output_dir, 'evaluation_results')}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    evaluate_model()
