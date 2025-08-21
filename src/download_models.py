import os
import mlflow
import argparse
from pathlib import Path


def download_models_from_experiment(experiment_name):
    """
    Download model_best.pth artifacts from all runs in a given MLflow experiment.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
    """
    # Set up MLflow tracking URI (same as in main.py)
    mlflow.set_tracking_uri("http://35.165.139.156:5000")
    
    # Always download to ../exp/single_pose/{experiment_name}
    output_dir = f"../exp/single_pose/{experiment_name}"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get the experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found!")
            return
        experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error accessing experiment '{experiment_name}': {e}")
        return
    
    print(f"Found experiment: {experiment_name} (ID: {experiment_id})")
    
    # Get all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    if runs.empty:
        print("No runs found in the experiment.")
        return
    
    print(f"Found {len(runs)} runs in the experiment.")
    
    # Iterate through each run
    for idx, run in runs.iterrows():
        run_id = run['run_id']
        run_name = run.get('tags.mlflow.runName', f"run_{run_id[:8]}")
        
        print(f"\nProcessing run: {run_name} (ID: {run_id})")
        
        try:
            # Create run-specific directory using run name
            run_dir = output_path / run_name
            run_dir.mkdir(exist_ok=True)
            
            # Download the model_best.pth artifact directly to the run directory
            artifact_path = f"runs:/{run_id}/model_best.pth"
            local_path = run_dir / "model_best.pth"
            
            print(f"Downloading model_best.pth to {local_path}")
            mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_path,
                dst_path=str(local_path)
            )
            
            # Check if file was downloaded successfully
            if local_path.exists():
                file_size = local_path.stat().st_size
                print(f"✓ Successfully downloaded model_best.pth ({file_size:,} bytes)")
            else:
                print("✗ Failed to download model_best.pth")
                
        except Exception as e:
            print(f"✗ Error downloading artifacts for run {run_name}: {e}")
            continue
    
    print(f"\nDownload complete! Models saved to: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Download model artifacts from MLflow experiment")
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True,
        help="Name of the MLflow experiment"
    )
    
    args = parser.parse_args()
    
    download_models_from_experiment(args.experiment)


if __name__ == "__main__":
    main()
