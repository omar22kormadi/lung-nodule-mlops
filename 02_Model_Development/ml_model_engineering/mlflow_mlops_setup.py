import os
import mlflow

# ── Paths ─────────────────────────────────────────────────────────────────────
# We use SQLite for the tracking backend (metrics/params/registry)
# and a local folder for the artifact backend (model files, PNGs, etc.)
BASE_DIR = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\02_Model_Development\ml_model_engineering"
DB_PATH  = os.path.join(BASE_DIR, "mlflow.db")
DB_URI   = f"sqlite:///{DB_PATH.replace(chr(92), '/')}"

# Set the Tracking URI to the Database (This solves the deprecation warning)
mlflow.set_tracking_uri(DB_URI)

def log_detector_experiment():
    print("--- Setting up Detector Experiment ---")
    # Create a dedicated experiment for the Detector
    mlflow.set_experiment("1_Lung_Nodule_Detection")
    
    with mlflow.start_run(run_name="YOLOv8m_Final_2.5D"):
        # 1. Log your architecture & hyperparameters
        mlflow.log_params({
            "model_type": "YOLOv8m",
            "input_type": "2.5D PNG",
            "optimizer": "SGD",
            "epochs": 50
        })
        
        # 2. Log your best metrics
        mlflow.log_metrics({
            "mAP_50": 0.708,
            "precision": 0.737,
            "recall": 0.681
        })
        
        # 3. Log the model file as an artifact (adjust path if needed)
        yolo_path = os.path.join(BASE_DIR, "models", "yolo_best.pt") # example path
        if os.path.exists(yolo_path):
            mlflow.log_artifact(yolo_path, artifact_path="model_weights")
            
        print("Detector logged successfully.")

def log_classifier_experiment():
    print("--- Setting up Classifier Experiment ---")
    # Create a dedicated experiment for the Classifier
    mlflow.set_experiment("2_Malignancy_Classification")
    
    with mlflow.start_run(run_name="R2Plus1D18_Final_Attention"):
        # 1. Log your architecture & hyperparameters
        mlflow.log_params({
            "model_type": "R(2+1)D-18",
            "attention": "Dual-Channel",
            "patch_size": "64x64x64",
            "optimizer": "AdamW"
        })
        
        # 2. Log your best metrics (from your desktop/paper metrics)
        mlflow.log_metrics({
            "roc_auc": 0.953,
            "val_accuracy": 0.881,
            "val_f1": 0.872
        })
        
        # 3. Log output files (like confusion matrix, roc curves from your desktop folder)
        classifier_output = r"C:\Users\amork\Desktop\classifier output"
        if os.path.exists(classifier_output):
            mlflow.log_artifacts(classifier_output, artifact_path="evaluation_plots")
            
        print("Classifier logged successfully.")

if __name__ == "__main__":
    print(f"Connecting to MLflow Database at: {DB_URI}\n")
    log_detector_experiment()
    print("")
    log_classifier_experiment()
    
    print("\n========================================================")
    print("DONE! To view your MLOps Dashboard, kill the old mlflow server, and run:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("========================================================")
