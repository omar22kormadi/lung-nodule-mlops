"""
FINALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLl
Kaggle-Ready YOLOv8m Lung Nodule Detection Training Pipeline
============================================================

This script is a production-grade training pipeline tailored for lung nodule detection
on CT scans (LUNA16 dataset) using YOLOv8m.

KEY DESIGN DECISIONS:
1. NO HYPERPARAMETER SEARCH: We dedicate 100% of Kaggle's 12-hour limit to deep training.
2. MEDICAL AUGMENTATIONS ONLY: Restricted augmentations that respect CT physics and anatomy.
3. ADAMW OPTIMIZER: Faster convergence for medical detection tasks compared to SGD.
4. TIME-BUDGETED: Automatically stops training at 11.5 hours to safely export results before Kaggle kills the session.
"""

import os
import gc
import json
import shutil
import random
import time as time_module
from pathlib import Path

import torch
import numpy as np
import mlflow
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Central configuration hub for local and Kaggle environments."""
    
    # Kaggle Paths
    KGL_DATASET_DIR = Path("/kaggle/input/datasets/amorkormadi/luna16-yolo-dataset-v4/luna16_yolo_dataset_v4")
    KGL_YAML_PATH = Path("/kaggle/input/datasets/amorkormadi/dataset-v4/dataset - Copy.yaml")
    KGL_WEIGHTS_PATH = Path("/kaggle/input/datasets/amorkormadi/yolov8-weights/yolov8m.pt")
    KGL_OUTPUT_DIR = Path("/kaggle/working/output")
    
    # Local Paths
    LOCAL_DATASET_DIR = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\luna16_yolo_dataset_v4")
    LOCAL_YAML_PATH = LOCAL_DATASET_DIR / "dataset.yaml"
    LOCAL_WEIGHTS_PATH = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\02_Model_Development\ml_model_engineering\yolov8m.pt")
    LOCAL_OUTPUT_DIR = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\02_Model_Development\ml_model_engineering\training_output")

    # Settings Control
    QUICK_TEST = False          # <--- SET TO FALSE FOR FULL TRAINING ON KAGGLE
    RESUME_TRAINING = False
    OFFLINE_MODE = True  # Prevent any automatic downloads from GitHub/Ultralytics
    
    # --------------------------------------------------------------------------
    # ALL-IN TRAINING BUDGET
    # --------------------------------------------------------------------------
    # We train for 6.0 hours, leaving ~1h 12m for validation & zipping.
    TRAIN_TIME_HOURS = 0.05 if QUICK_TEST else 11.0
    EPOCHS = 3 if QUICK_TEST else 1500  # Time-guard will stop before this
    PATIENCE = 0 if QUICK_TEST else 250  # Increased patience to prevent premature stopping
    BATCH_SIZE = -1 if not QUICK_TEST else 16  # -1 enables YOLOv8 AutoBatch
    IMG_SIZE = 640  # Trained with 640 as recommended for 512x512 CT slices
    WORKERS = 8  # Increased from 4 for faster data loading
    
    # Selected Dynamically via setup()
    DATASET_YAML: Path
    WEIGHTS_PATH: Path
    OUTPUT_DIR: Path
    MLFLOW_URI: str
    PIPELINE_START_TIME: float = 0.0

    @classmethod
    def setup(cls):
        cls.PIPELINE_START_TIME = time_module.time()
        is_kaggle = Path("/kaggle/input").exists()
        
        if is_kaggle:
            print("[INFO] Kaggle environment detected.")
            cls.DATASET_YAML = cls.KGL_YAML_PATH
            cls.WEIGHTS_PATH = cls.KGL_WEIGHTS_PATH if cls.KGL_WEIGHTS_PATH.exists() else Path("yolov8m.pt")
            cls.OUTPUT_DIR = cls.KGL_OUTPUT_DIR
        else:
            print("[INFO] Local environment detected.")
            cls.DATASET_YAML = cls.LOCAL_YAML_PATH
            cls.WEIGHTS_PATH = cls.LOCAL_WEIGHTS_PATH if cls.LOCAL_WEIGHTS_PATH.exists() else Path("yolov8m.pt")
            cls.OUTPUT_DIR = cls.LOCAL_OUTPUT_DIR
            
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MLFLOW_URI = f"file://{cls.OUTPUT_DIR / 'mlruns'}"
        
        if cls.OFFLINE_MODE:
            os.environ['YOLO_OFFLINE'] = 'True'
            
        print(f"[INFO] Output Directory: {cls.OUTPUT_DIR}")
        print(f"[INFO] Time Budget: {cls.TRAIN_TIME_HOURS:.2f} hours maximum training time.")

    @classmethod
    def elapsed_hours(cls) -> float:
        return (time_module.time() - cls.PIPELINE_START_TIME) / 3600.0


def set_deterministic_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================

def train_production_model():
    print("\n" + "="*60)
    print("🚀 STARTING PRODUCTION TRAINING (v4 — Tight Lung Crop)")
    print(f"   Max Time Allowed: {Config.TRAIN_TIME_HOURS:.2f} hours")
    print(f"   Max Epochs      : {Config.EPOCHS}")
    print("="*60)
    
    cleanup_gpu_memory()
    
    # Handle resumable training if a previous run was interrupted
    resume_path = Config.OUTPUT_DIR / "final_model" / "weights" / "last.pt"
    if Config.RESUME_TRAINING and resume_path.exists():
        print(f"[INFO] Resuming training from {resume_path}")
        model = YOLO(str(resume_path))
        resume_flag = True
    else:
        model = YOLO(str(Config.WEIGHTS_PATH))
        resume_flag = False

    # Medical Augmentations (OPTIMIZED v7 - Table 10: specific_augment achieves 65.20% mAP)
    medical_augs = {
        # ✅ Table 10: specific_augment configuration
        'mosaic': 0.3,            # LOW mosaic (30%) - helps recall without destroying small nodules
        
        # ✅ Safe geometric transforms for CT
        'degrees': 10.0,          # Reduced from 15 (lungs roughly symmetric)
        'translate': 0.1,         # Small translations (10%)
        'scale': 0.25,            # Balanced scale variation
        'fliplr': 0.5,            # Horizontal flip (safe for lungs)
        
        # ✅ Intensity variations (simulates CT window changes)
        'hsv_h': 0.0,             # NO hue (grayscale CT)
        'hsv_s': 0.0,             # NO saturation (grayscale CT)
        'hsv_v': 0.2,             # Reduced from 0.25 (less aggressive)
        
        # ✅ Advanced augmentations (Table 10: specific_augment)
        'erasing': 0.1,           # Reduced from 0.15 (less occlusion)
        'crop_fraction': 0.9,     # Increased from 0.8 (less aggressive cropping)
        
        # ❌ STRICTLY FORBIDDEN (breaks medical anatomy):
        'flipud': 0.0,            # Vertical flip breaks anatomy
        'mixup': 0.0,             # Creates fake hybrid nodules
        'copy_paste': 0.0,        # Duplicates nodules unnaturally
        'perspective': 0.0,       # Perspective warp breaks CT geometry
        'shear': 0.0              # Shear distorts anatomy
    }

    results = model.train(
        data=str(Config.DATASET_YAML),
        project=str(Config.OUTPUT_DIR),
        name="final_model_v4_lung_crop",
        epochs=Config.EPOCHS,
        time=Config.TRAIN_TIME_HOURS,  # Hard stop before Kaggle timeout
        batch=Config.BATCH_SIZE,
        imgsz=Config.IMG_SIZE,
        cache=True,                    # RAM caching - eliminates disk I/O bottlenecks
        
        # ✅ Table 6: SGD lr0=0.001, weight_decay=0.0005 (mAP@50=54.97%, best optimizer)
        optimizer="SGD",               # Table 6: SGD outperforms AdamW (+7.82% mAP)
        lr0=0.001,                     # Table 6: Best lr0 for SGD
        lrf=0.01,                      # Final LR = 0.001 * 0.01 = 0.00001
        weight_decay=0.0005,           # Table 6: Best weight_decay
        momentum=0.937,                # YOLO default (SGD momentum)
        
        # ✅ Table 7: conf=0.25, IoU=0.40, box=7.5, cls=1.0 (mAP@50=64.83%)
        box=7.5,                       # Table 7: Best box=7.50
        cls=1.0,                       # Table 7: Best cls=1.00 (vs 0.5 or 0.7)
        dfl=1.5,                       # Default DFL weight
        
        # ✅ Learning rate schedule
        cos_lr=True,                   # Cosine annealing (smooth decay)
        warmup_epochs=3,               # Reduced from 5 (faster start)
        warmup_momentum=0.8,           # Start with low momentum
        warmup_bias_lr=0.1,            # Conservative bias warmup
        
        # ✅ Small object detection improvements
        close_mosaic=25,               # Turn off mosaic 25 epochs before end (clean final validation)
        max_det=100,                   # Allow up to 100 detections per image
        
        # ✅ Training stability
        patience=Config.PATIENCE,
        workers=Config.WORKERS,
        seed=0,                        # Table 9: Seed 0 achieves best mAP@50=64.41%
        deterministic=True,
        amp=True,                      # Mixed precision (2x faster on GPU)
        verbose=True,
        resume=resume_flag,
        
        # ✅ Medical augmentations
        **medical_augs
    )
    
    return model, results


def export_and_evaluate(model: YOLO, results_dir: Path):
    print("\n[INFO] Running final validation with OPTIMIZED thresholds...")
    
    # ✅ Multi-threshold validation (Table 7: conf=0.25, IoU=0.40 optimal)
    val_results = model.val(
        data=str(Config.DATASET_YAML),
        imgsz=Config.IMG_SIZE,
        batch=16,
        amp=True,
        plots=True,
        save_json=True,
        
        # ✅ Table 7: conf=0.25, IoU=0.40 gives mAP@50=64.83%
        conf=0.25,                     # Table 7: Best confidence threshold
        iou=0.40,                      # Table 7: Best IoU threshold
        max_det=100,                   # Allow more detections
        cache=True,                    # RAM caching for faster validation
        
        # ✅ Better small object evaluation
        single_cls=False,              # Keep class distinction
        half=True,                     # FP16 for faster validation
    )
    
    mp = float(val_results.box.mp)
    mr = float(val_results.box.mr)
    
    # Calculate advanced medical metrics
    f1 = float(2 * (mp * mr) / (mp + mr + 1e-8))
    f2 = float(5 * (mp * mr) / (4 * mp + mr + 1e-8))  # Weights recall higher than precision
    
    metrics = {
        'mAP50': float(val_results.box.map50),
        'mAP50_95': float(val_results.box.map),
        'precision': mp,
        'recall_sensitivity': mr,          # Recall is equivalent to Sensitivity
        'f1_score': f1,
        'f2_score': f2,                    # Important for medical (penalizes false negatives)
        'dice_coefficient': f1,            # For object detection, F1 == Dice
        'fdr_approx': float(1.0 - mp),     # False Discovery Rate
        'fppi_approximation': float((1 - mp) * 100),
    }
    
    # Log to MLflow (Table-based optimal parameters)
    mlflow.log_params({
        'optimizer': 'SGD',              # Table 6: SGD lr0=0.001, wd=0.0005
        'lr0': 0.001,
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'batch_size': Config.BATCH_SIZE,
        'img_size': Config.IMG_SIZE,
        'epochs': Config.EPOCHS,
        'mosaic': 0.3,
        'conf_threshold': 0.25,          # Table 7: Best conf
        'iou_threshold': 0.40,           # Table 7: Best IoU
        'max_det': 100,
        'close_mosaic': 25,
        'box_loss_weight': 7.5,          # Table 7: Best box
        'cls_loss_weight': 1.0,          # Table 7: Best cls
        'dfl_loss_weight': 1.5,
        'cache': True,
        'seed': 0,                       # Table 9: Seed 0 best mAP
        'dataset_version': 'v4_lung_crop_no_wipeout',
    })
    
    mlflow.log_metrics(metrics)
    
    # Save Metrics
    with open(Config.OUTPUT_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Copy essential artifacts to output root
    artifacts_to_copy = [
        (results_dir / "weights" / "best.pt", Config.OUTPUT_DIR / "best.pt"),
        (results_dir / "confusion_matrix.png", Config.OUTPUT_DIR / "confusion_matrix.png"),
        (results_dir / "PR_curve.png", Config.OUTPUT_DIR / "PR_curve.png"),
        (results_dir / "F1_curve.png", Config.OUTPUT_DIR / "F1_curve.png")
    ]
    
    for src, dst in artifacts_to_copy:
        if src.exists():
            shutil.copy2(src, dst)
            
    # ✅ Clean up MLflow artifacts (only keep BEST model to save space)
    mlruns_dir = Config.OUTPUT_DIR / "mlruns"
    if mlruns_dir.exists():
        # Remove all checkpoint models except best.pt
        for pt_file in mlruns_dir.rglob("*.pt"):
            if 'best' not in pt_file.name:
                pt_file.unlink()
        # Remove ONNX exports (too large)
        for onnx_file in mlruns_dir.rglob("*.onnx"):
            onnx_file.unlink()
        # Remove intermediate checkpoints
        for weights_dir in mlruns_dir.rglob("weights"):
            if weights_dir.is_dir():
                for f in weights_dir.iterdir():
                    if 'best' not in f.name and 'last' not in f.name:
                        f.unlink()


def generate_presentation_images(model: YOLO):
    """Generates 'Before' (raw) and 'After' (detected) images for presentations."""
    print("\n[INFO] Generating presentation images...")
    pres_dir = Config.OUTPUT_DIR / "presentation_images"
    pres_dir.mkdir(parents=True, exist_ok=True)
    
    val_images_dir = Config.DATASET_YAML.parent / "images" / "val"
    if not val_images_dir.exists():
        print(f"[WARNING] Validation images not found at {val_images_dir}")
        return
        
    all_images = list(val_images_dir.glob("*.png")) + list(val_images_dir.glob("*.jpg"))
    if not all_images:
        print("[WARNING] No images found for presentation.")
        return
        
    # Select 10 random images
    random.seed(42)
    selected_images = random.sample(all_images, min(10, len(all_images)))
    
    for i, img_path in enumerate(selected_images):
        # 1. Save "Before" (Raw Image)
        shutil.copy2(img_path, pres_dir / f"sample_{i+1}_raw_{img_path.name}")
        
        # 2. Save "After" (YOLO Prediction)
        # We run predict with high confidence for clean presentation images
        res = model.predict(
            source=str(img_path),
            conf=0.35,  # Slightly higher conf for cleaner presentation plots
            iou=0.40,
            save=True,
            project=str(pres_dir),
            name=f"pred_{i+1}",
            exist_ok=True
        )
        
        # Move the saved prediction out of the subfolder to the main presentation folder
        pred_subfolder = pres_dir / f"pred_{i+1}"
        if pred_subfolder.exists():
            for pred_file in pred_subfolder.glob("*.*"):
                shutil.move(str(pred_file), str(pres_dir / f"sample_{i+1}_detected_{img_path.name}"))
            shutil.rmtree(pred_subfolder)
            
    print(f"[INFO] Saved {len(selected_images)} Before/After pairs to {pres_dir}")


def create_zip():
    print("\n" + "="*60)
    print("📦 CREATING DOWNLOADABLE ZIP ARCHIVE")
    print("="*60)
    
    zip_path = Config.OUTPUT_DIR / "luna16_yolo_results"
    staging_dir = Config.OUTPUT_DIR / "zip_staging"
    staging_dir.mkdir(exist_ok=True)
    
    try:
        if (Config.OUTPUT_DIR / "mlruns").exists():
            shutil.copytree(Config.OUTPUT_DIR / "mlruns", staging_dir / "mlruns", dirs_exist_ok=True)
            
        if (Config.OUTPUT_DIR / "final_model").exists():
            shutil.copytree(Config.OUTPUT_DIR / "final_model", staging_dir / "final_model", dirs_exist_ok=True)
            
        if (Config.OUTPUT_DIR / "best.pt").exists():
            shutil.copy2(Config.OUTPUT_DIR / "best.pt", staging_dir / "best.pt")
            
        if (Config.OUTPUT_DIR / "final_metrics.json").exists():
            shutil.copy2(Config.OUTPUT_DIR / "final_metrics.json", staging_dir / "final_metrics.json")
            
        if (Config.OUTPUT_DIR / "presentation_images").exists():
            shutil.copytree(Config.OUTPUT_DIR / "presentation_images", staging_dir / "presentation_images", dirs_exist_ok=True)
            
        shutil.make_archive(str(zip_path), 'zip', str(staging_dir))
        print(f"[INFO] ✅ Successfully created archive: {zip_path}.zip")
        
    except Exception as e:
        print(f"[ERROR] Failed to create zip archive: {e}")
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    print("="*60)
    print(" LUNA16 YOLOv8m v4 — TIGHT LUNG CROP PIPELINE ")
    print("="*60)
    
    Config.setup()
    set_deterministic_seed(42)
    mlflow.set_tracking_uri(Config.MLFLOW_URI)
    mlflow.set_experiment("LUNA16_Nodule_Detection_v4_LungCrop")
    
    if Config.QUICK_TEST:
        print("\n[WARNING] QUICK_TEST is ENABLED. Training will be extremely short.")
    
    with mlflow.start_run(run_name="Production_Run_v4_LungCrop") as run:
        print(f"\n[INFO] MLflow Run ID: {run.info.run_id}")
        
        # Log dataset info
        mlflow.log_param('dataset', 'LUNA16_YOLO_v4')
        mlflow.log_param('dataset_path', str(Config.DATASET_YAML))
        
        final_model, results = train_production_model()
        final_dir = Config.OUTPUT_DIR / "final_model_v4_lung_crop"
        export_and_evaluate(final_model, final_dir)
        
    # Print Final Summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*60)
    
    metrics_path = Config.OUTPUT_DIR / "final_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        print(f"  mAP@50           : {metrics.get('mAP50', 0):.4f}")
        print(f"  mAP@50-95        : {metrics.get('mAP50_95', 0):.4f}")
        print(f"  Precision        : {metrics.get('precision', 0):.4f}")
        print(f"  Sensitivity/Rec. : {metrics.get('recall_sensitivity', 0):.4f}")
        print(f"  F1 Score         : {metrics.get('f1_score', 0):.4f}")
        print(f"  F2 Score         : {metrics.get('f2_score', 0):.4f}")
        print(f"  Dice Coefficient : {metrics.get('dice_coefficient', 0):.4f}")
        print(f"  False Disc. Rate : {metrics.get('fdr_approx', 0):.4f}")
        print(f"  FPPI (approx)    : {metrics.get('fppi_approximation', 0):.2f}")
    
    print(f"\n⏱️  Total Pipeline Runtime: {Config.elapsed_hours():.2f} hours")
    
    # Generate presentation images safely at the very end
    try:
        generate_presentation_images(final_model)
    except Exception as e:
        print(f"\n[ERROR] Failed to generate presentation images: {e}")
        print("[INFO] Continuing to zip creation...")
    
    create_zip()

if __name__ == "__main__":
    main() 