import os
import mlflow
import pandas as pd

# Set the tracking URI to the mlruns folder
# We use the absolute path to ensure MLflow finds it
MLRUNS_DIR = r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\02_Model_Development\ml_model_engineering\mlruns"
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")

def explore_mlflow_runs():
    print("=" * 60)
    print("MLflow Experiment Explorer")
    print("=" * 60)
    
    # Get all experiments
    experiments = mlflow.search_experiments()
    
    if not experiments:
        print("No experiments found in mlruns directory.")
        return

    print(f"Found {len(experiments)} experiments.\n")
    
    for exp in experiments:
        print("-" * 60)
        print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
        print(f"Artifact Location: {exp.artifact_location}")
        
        # Search for runs in this experiment
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        
        if runs.empty:
            print("  No runs logged for this experiment.")
            continue
            
        print(f"  Total runs: {len(runs)}")
        
        # Display the top 3 runs based on a primary metric if available
        # Let's try to sort by 'metrics.val_roc_auc' or 'metrics.mAP50' if they exist
        sort_col = None
        if 'metrics.val_roc_auc' in runs.columns:
            sort_col = 'metrics.val_roc_auc'
            ascending = False
        elif 'metrics.mAP50' in runs.columns:
            sort_col = 'metrics.mAP50'
            ascending = False
        
        if sort_col:
            runs = runs.sort_values(by=sort_col, ascending=ascending)
            print(f"  Top 3 runs sorted by {sort_col}:")
        else:
            print("  Showing recent runs:")
            
        for idx, row in runs.head(3).iterrows():
            print(f"    - Run ID: {row['run_id']}")
            print(f"      Status: {row['status']}")
            
            # Print parameters safely
            params = {k: v for k, v in row.items() if k.startswith('params.') and pd.notna(v)}
            if params:
                print("      Key Parameters:")
                # Show first 3 params
                for k, v in list(params.items())[:3]:
                    print(f"        {k.replace('params.', '')}: {v}")
            
            # Print metrics safely
            metrics = {k: v for k, v in row.items() if k.startswith('metrics.') and pd.notna(v)}
            if metrics:
                print("      Metrics:")
                for k, v in metrics.items():
                    print(f"        {k.replace('metrics.', '')}: {v:.4f}")
            print()

if __name__ == "__main__":
    explore_mlflow_runs()
