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
            # First check if model_best.pth artifact exists in the run
            try:
                client = mlflow.tracking.MlflowClient()
                artifacts = client.list_artifacts(run_id)
                artifact_names = [artifact.path for artifact in artifacts]
                
                if "model_best.pth" not in artifact_names:
                    print(f"⚠ Run {run_name} does not have model_best.pth artifact - skipping")
                    continue
                    
            except Exception as artifact_check_error:
                print(f"⚠ Could not check artifacts for run {run_name}: {artifact_check_error} - skipping")
                continue
            
            # Create run-specific directory using run name
            run_dir = output_path / run_name
            run_dir.mkdir(exist_ok=True)
            
            # Check if model_best.pth already exists and is a file
            local_path = run_dir / "model_best.pth"
            if local_path.exists() and local_path.is_file():
                file_size = local_path.stat().st_size
                if file_size > 1000000:  # More than 1MB
                    print(f"✓ Model already exists: {local_path} ({file_size:,} bytes)")
                    continue
                else:
                    print(f"⚠ Existing model file is too small ({file_size:,} bytes), re-downloading...")
                    local_path.unlink()
            
            # Remove directory if it exists (from previous failed download)
            if local_path.exists() and local_path.is_dir():
                print(f"⚠ Removing existing directory: {local_path}")
                import shutil
                shutil.rmtree(local_path)
            
            # Download the model_best.pth artifact
            print(f"Downloading model_best.pth to {local_path}")
            
            # Use mlflow.artifacts.download_artifacts with proper error handling
            try:
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{run_id}/model_best.pth",
                    dst_path=str(run_dir)
                )
                
                # Verify the download
                if local_path.exists() and local_path.is_file():
                    file_size = local_path.stat().st_size
                    if file_size > 1000000:  # More than 1MB
                        print(f"✓ Successfully downloaded model_best.pth ({file_size:,} bytes)")
                    else:
                        print(f"⚠ Downloaded file is too small ({file_size:,} bytes) - may be corrupted")
                else:
                    print(f"✗ Failed to download model_best.pth - file not found")
                    
            except Exception as download_error:
                print(f"✗ Error downloading model_best.pth: {download_error}")
                
                # Try alternative download method
                print("Trying alternative download method...")
                try:
                    client = mlflow.tracking.MlflowClient()
                    client.download_artifacts(
                        run_id=run_id,
                        path="model_best.pth",
                        dst_path=str(run_dir)
                    )
                    
                    if local_path.exists() and local_path.is_file():
                        file_size = local_path.stat().st_size
                        if file_size > 1000000:
                            print(f"✓ Successfully downloaded model_best.pth using alternative method ({file_size:,} bytes)")
                        else:
                            print(f"⚠ Downloaded file is too small ({file_size:,} bytes)")
                    else:
                        print(f"✗ Alternative download method also failed")
                        
                except Exception as alt_error:
                    print(f"✗ Alternative download method failed: {alt_error}")
                
        except Exception as e:
            print(f"✗ Error processing run {run_name}: {e}")
            continue
    
    print(f"\nDownload complete! Models saved to: {output_path.absolute()}")
    
    # Summary of downloaded models
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY:")
    print("="*60)
    
    for run_dir in output_path.iterdir():
        if run_dir.is_dir():
            model_file = run_dir / "model_best.pth"
            if model_file.exists() and model_file.is_file():
                file_size = model_file.stat().st_size
                print(f"✓ {run_dir.name}: {file_size:,} bytes")
            else:
                print(f"✗ {run_dir.name}: No model file found")


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
