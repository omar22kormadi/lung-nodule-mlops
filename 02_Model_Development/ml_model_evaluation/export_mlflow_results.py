"""
Export all MLflow experiment results to a text file, sorted worst to best
"""
import mlflow
from pathlib import Path
import datetime

# Set MLflow tracking URI - mlruns is in ml_model_engineering
ml_model_eval_path = Path(__file__).parent
ml_model_eng_path = ml_model_eval_path.parent / "ml_model_engineering"
mlruns_path = ml_model_eng_path / "mlruns"
mlflow.set_tracking_uri(f"file:///{mlruns_path.absolute()}")

# Output file - save to current directory (ml_model_evaluation)
output_file = Path(__file__).parent / "mlflow_all_results.txt"

print("Extracting all MLflow results...")

# Get all experiments
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

all_runs_data = []

for exp in experiments:
    if exp.name in [".trash", "Default", "mlruns"]:
        continue
    
    try:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id]
        )
        
        for run in runs:
            run_info = {
                'experiment_name': exp.name,
                'run_id': run.info.run_id,
                'status': run.info.status,
                'start_time': datetime.datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': run.data.metrics,
                'params': run.data.params,
            }
            all_runs_data.append(run_info)
    except Exception as e:
        print(f"Error in experiment {exp.name}: {e}")

# Sort by val_accuracy (best to worst - descending order)
# If val_accuracy not available, try other metrics
def get_sort_key(run):
    metrics = run['metrics']
    # Priority order for sorting
    if 'val_accuracy' in metrics:
        return metrics['val_accuracy']
    elif 'final_metrics_mAP50_B' in metrics:
        return metrics['final_metrics_mAP50_B']
    elif 'val_auc' in metrics:
        return metrics['val_auc']
    elif 'train_accuracy' in metrics:
        return metrics['train_accuracy']
    else:
        return 0.0  # Unknown metrics go last

all_runs_data.sort(key=get_sort_key, reverse=True)  # reverse=True for best to worst

# Write to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("MLFLOW EXPERIMENT RESULTS - COMPLETE SUMMARY\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Runs: {len(all_runs_data)}\n")
    f.write(f"Sorted: Best to Worst (by validation accuracy/mAP50)\n")
    f.write("=" * 100 + "\n\n")
    
    for idx, run in enumerate(all_runs_data, 1):
        f.write(f"\n{'='*100}\n")
        f.write(f"RANK #{idx} (Score: {get_sort_key(run):.4f})\n")
        f.write(f"{'='*100}\n")
        
        f.write(f"\nExperiment: {run['experiment_name']}\n")
        f.write(f"Run ID: {run['run_id']}\n")
        f.write(f"Status: {run['status']}\n")
        f.write(f"Start Time: {run['start_time']}\n")
        
        # Parameters
        f.write(f"\n--- PARAMETERS ---\n")
        if run['params']:
            for param_name, param_value in sorted(run['params'].items()):
                f.write(f"  {param_name}: {param_value}\n")
        else:
            f.write(f"  (No parameters logged)\n")
        
        # Metrics
        f.write(f"\n--- METRICS ---\n")
        if run['metrics']:
            for metric_name, metric_value in sorted(run['metrics'].items()):
                f.write(f"  {metric_name}: {metric_value:.4f}\n")
        else:
            f.write(f"  (No metrics logged)\n")
        
        f.write(f"\n")
    
    # Summary table at the end
    f.write(f"\n{'='*100}\n")
    f.write("SUMMARY TABLE - ALL RUNS\n")
    f.write(f"{'='*100}\n")
    
    f.write(f"\n{'Rank':<5} {'Experiment':<45} {'Run ID':<10} {'Val Acc/mAP':<12} {'AUC':<10} {'F1':<10}\n")
    f.write(f"{'-'*5} {'-'*45} {'-'*10} {'-'*12} {'-'*10} {'-'*10}\n")
    
    for idx, run in enumerate(all_runs_data, 1):
        exp_name = run['experiment_name'][:44]
        run_id = run['run_id'][:8]
        
        val_acc = run['metrics'].get('val_accuracy', 
                  run['metrics'].get('final_metrics_mAP50_B', 
                  run['metrics'].get('train_accuracy', 0.0)))
        auc = run['metrics'].get('val_auc', 0.0)
        f1 = run['metrics'].get('val_f1', 0.0)
        
        f.write(f"{idx:<5} {exp_name:<45} {run_id:<10} {val_acc:<12.4f} {auc:<10.4f} {f1:<10.4f}\n")
    
    f.write(f"\n{'='*100}\n")
    f.write("END OF REPORT\n")
    f.write(f"{'='*100}\n")

print(f"\n✅ Results exported to: {output_file}")
print(f"📊 Total runs processed: {len(all_runs_data)}")
print(f"📈 Best run: Rank #{len(all_runs_data)} (Score: {get_sort_key(all_runs_data[-1]):.4f})")
print(f"📉 Worst run: Rank #1 (Score: {get_sort_key(all_runs_data[0]):.4f})")
