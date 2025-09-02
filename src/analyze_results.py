from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
import argparse
import csv

def download_metric_data(experiment_id, metric_name="val/mAP0.50:0.95"):
    """
    Download metric data for all runs in an experiment to CSV files.
    
    Args:
        experiment_id (str): The MLflow experiment ID
        metric_name (str): The metric name to download
    """
    
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri("http://35.165.139.156:5000")
    
    try:
        # Get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_id)
        if experiment is None:
            print(f"Error: Experiment '{experiment_id}' not found")
            return
        
        print(f"Downloading metric data for experiment: {experiment_id}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print("=" * 80)
        
        # Get all runs for this experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            print("No runs found for this experiment")
            return
        
        print(f"Found {len(runs)} runs")
        print()
        
        # Create output directory
        output_dir = Path(f"exp_metrics/{experiment_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download metric data for each run
        for idx, run in runs.iterrows():
            run_id = run['run_id']
            run_name = run['tags.mlflow.runName'] if 'tags.mlflow.runName' in run else run_id
            
            print(f"Downloading data for run: {run_name}")
            print(f"Run ID: {run_id}")
            
            try:
                client = MlflowClient()

                # Your run ID and metric key
                metric_key = metric_name

                # Get metric history
                history = client.get_metric_history(run_id, metric_key)

                # Convert to DataFrame
                df = pd.DataFrame(
                    [(m.step, m.value, m.timestamp) for m in history],
                    columns=["step", "value", "timestamp"]
                )

                print(df.head())
                df.to_csv(f"{metric_name}.csv", index=False)

                metrics = client.get_metric_history(run_id, metric_key)
                
                if metrics:
                    # Create CSV file
                    csv_file = output_dir / f"{run_name}.csv"
                    
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['step', 'value', 'timestamp'])
                        
                        for metric in metrics:
                            writer.writerow([metric.step, metric.value, metric.timestamp])
                    
                    print(f"  Downloaded {len(metrics)} data points to {csv_file}")
                else:
                    print(f"  No metric data found for {metric_name}")
                
            except Exception as e:
                print(f"  Error downloading data for run {run_name}: {e}")
            
            print()
        
        print(f"Download complete! Data saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error downloading experiment data: {e}")
        import traceback
        traceback.print_exc()

def analyze_mlflow_experiment(experiment_id):
    """
    Analyze MLflow experiment results by reading downloaded CSV files.
    
    Args:
        experiment_id (str): The MLflow experiment ID to analyze
    """
    
    # Path to downloaded data
    data_dir = Path(f"exp_metrics/{experiment_id}")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please run download_metric_data first.")
        return
    
    print(f"Analyzing experiment: {experiment_id}")
    print(f"Data directory: {data_dir}")
    print("=" * 80)
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in data directory")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print()
    
    # Analyze each run
    run_results = []
    
    for csv_file in csv_files:
        run_name = csv_file.stem  # filename without extension
        
        print(f"Analyzing run: {run_name}")
        print(f"CSV file: {csv_file}")
        
        try:
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            if not df.empty:
                # Find maximum value and its step
                max_idx = df['value'].idxmax()
                max_mAP = df.loc[max_idx, 'value']
                max_epoch = df.loc[max_idx, 'step']
                total_epochs = len(df)
                
                print(f"  Max mAP: {max_mAP:.4f} at epoch {max_epoch}")
                print(f"  Total epochs with metrics: {total_epochs}")
                
                run_results.append({
                    'run_name': run_name,
                    'run_id': csv_file.name,  # Use filename as run_id
                    'max_mAP': max_mAP,
                    'max_epoch': max_epoch,
                    'status': 'Success'
                })
            else:
                print(f"  No data found in CSV file")
                run_results.append({
                    'run_name': run_name,
                    'run_id': csv_file.name,
                    'max_mAP': 0.0,
                    'max_epoch': 0,
                    'status': 'No data'
                })
                
        except Exception as e:
            print(f"  Error analyzing CSV file: {e}")
            run_results.append({
                'run_name': run_name,
                'run_id': csv_file.name,
                'max_mAP': 0.0,
                'max_epoch': 0,
                'status': f'Error: {str(e)}'
            })
        
        print()
    
    # Sort results by max mAP in descending order
    run_results.sort(key=lambda x: x['max_mAP'], reverse=True)
    
    # Print summary
    print("=" * 80)
    print("SUMMARY - Runs sorted by maximum validation mAP (descending)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Run Name':<40} {'Max mAP':<10} {'Epoch':<8} {'Status':<15}")
    print("-" * 80)
    
    for rank, result in enumerate(run_results, 1):
        if result['status'] == 'Success':
            print(f"{rank:<5} {result['run_name']:<40} {result['max_mAP']:<10.4f} {result['max_epoch']:<8} {result['status']:<15}")
        else:
            print(f"{rank:<5} {result['run_name']:<40} {'N/A':<10} {'N/A':<8} {result['status']:<15}")
    
    print("=" * 80)
    
    # Print top 5 runs
    successful_runs = [r for r in run_results if r['status'] == 'Success']
    if successful_runs:
        print("\n🏆 TOP 5 RUNS:")
        print("-" * 80)
        for rank, result in enumerate(successful_runs[:5], 1):
            print(f"{rank}. {result['run_name']}")
            print(f"   Max mAP: {result['max_mAP']:.4f} at epoch {result['max_epoch']}")
            print(f"   CSV file: {result['run_id']}")
            print()
    
    # Save results to CSV
    df = pd.DataFrame(run_results)
    output_file = f"mlflow_analysis_{experiment_id.replace('/', '_')}.csv"
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Download and analyze MLflow experiment results')
    parser.add_argument('--exp_id', type=str, required=True,
                       help='MLflow experiment ID to analyze')
    parser.add_argument('--metric', type=str, default='val/mAP0.50:0.95',
                       help='Metric name to download (default: val/mAP0.50:0.95)')
    
    args = parser.parse_args()
    
    download_metric_data(args.exp_id, args.metric)
    
    analyze_mlflow_experiment(args.exp_id)

if __name__ == '__main__':
    main()
