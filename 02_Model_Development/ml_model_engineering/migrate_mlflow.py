import os
from mlflow.tracking import MlflowClient
from mlflow.entities import Param, Metric, RunTag

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\02_Model_Development\ml_model_engineering"

SRC_DETECTOR_URI   = f"file:///{os.path.join(BASE_DIR, 'mlruns').replace(chr(92), '/')}"
SRC_CLASSIFIER_URI = "sqlite:///C:/Users/amork/Desktop/kaggle/mlflow_classifier.db"

# Create a brand new unified DB so we don't mess up anything
DEST_DB_PATH = os.path.join(BASE_DIR, "mlflow_unified.db")
DEST_URI     = f"sqlite:///{DEST_DB_PATH.replace(chr(92), '/')}"

def migrate(src_uri, dest_client, prefix_exp_name=""):
    try:
        src_client = MlflowClient(tracking_uri=src_uri)
        experiments = src_client.search_experiments()
    except Exception as e:
        print(f"Failed to connect to source {src_uri}: {e}")
        return

    for exp in experiments:
        # We wrap in try-except because some local YOLO tracking folders are corrupted/malformed
        try:
            runs = src_client.search_runs(exp.experiment_id)
        except Exception as e:
            print(f"  -> [WARNING] Skipping experiment '{exp.name}' due to parse error (corrupted folder).")
            continue
            
        if len(runs) == 0:
            continue
            
        # Give a clean name, even if it was originally 'Default'
        base_name = exp.name if exp.name.lower() != "default" else "Main_Runs"
        dest_exp_name = prefix_exp_name + base_name
        
        # Create experiment in destination
        try:
            dest_exp_id = dest_client.create_experiment(dest_exp_name)
        except Exception:
            dest_exp = dest_client.get_experiment_by_name(dest_exp_name)
            if dest_exp:
                dest_exp_id = dest_exp.experiment_id
            else:
                print(f"Could not create or find experiment {dest_exp_name}")
                continue
            
        print(f"  -> Migrating {len(runs)} runs from '{exp.name}' into unified '{dest_exp_name}'...")
        
        for run in runs:
            # Create a new run in the destination
            dest_run = dest_client.create_run(dest_exp_id, start_time=run.info.start_time)
            
            # 1. Log Params
            params = [Param(k, str(v)[:250]) for k, v in run.data.params.items()]
            if params:
                dest_client.log_batch(dest_run.info.run_id, params=params)
                
            # 2. Log Metrics (fetch full history so curves work!)
            for metric_key in run.data.metrics.keys():
                try:
                    metric_history = src_client.get_metric_history(run.info.run_id, metric_key)
                    chunk_size = 500
                    for i in range(0, len(metric_history), chunk_size):
                        chunk = metric_history[i:i+chunk_size]
                        dest_client.log_batch(dest_run.info.run_id, metrics=chunk)
                except Exception as e:
                    pass # Ignore if metric history is unreadable
                    
            # 3. Log Tags (this includes run name!)
            tags = [RunTag(k, str(v)) for k, v in run.data.tags.items()]
            if tags:
                dest_client.log_batch(dest_run.info.run_id, tags=tags)
                
            # Finish run with original status/time
            dest_client.set_terminated(dest_run.info.run_id, status=run.info.status, end_time=run.info.end_time)

if __name__ == "__main__":
    print("========================================================")
    print("  MLflow Unified Database Migration Script")
    print("========================================================")
    
    # Remove old unified DB to start fresh
    if os.path.exists(DEST_DB_PATH):
        try:
            os.remove(DEST_DB_PATH)
        except:
            pass
        
    print(f"Setting up unified Database at: {DEST_URI}\n")
    dest_client = MlflowClient(tracking_uri=DEST_URI)
    
    print("--- Migrating Detector Runs (Local Folder) ---")
    if os.path.exists(os.path.join(BASE_DIR, 'mlruns')):
        migrate(SRC_DETECTOR_URI, dest_client, prefix_exp_name="Detector_")
    else:
        print("  No local 'mlruns' folder found.")
        
    print("\n--- Migrating Classifier Runs (Kaggle DB) ---")
    if os.path.exists(r"C:\Users\amork\Desktop\kaggle\mlflow_classifier.db"):
        migrate(SRC_CLASSIFIER_URI, dest_client, prefix_exp_name="Classifier_")
    else:
        print("  Kaggle DB not found at the specified path.")
    
    print("\n========================================================")
    print("MIGRATION COMPLETE!")
    print("Start your unified MLOps Dashboard with:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow_unified.db --port 5000")
    print("========================================================")
