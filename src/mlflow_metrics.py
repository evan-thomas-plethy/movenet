import mlflow
import pandas as pd

def print_top_n_lowest_val_loss(experiment_name_or_id, metric_name="metrics.val/loss", top_n=10):
    mlflow.set_experiment(experiment_name_or_id)

    experiment = mlflow.get_experiment_by_name(experiment_name_or_id)
    if experiment is None:
        experiment = mlflow.get_experiment(experiment_name_or_id)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name_or_id}' not found.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if metric_name not in runs.columns:
        raise ValueError(f"Metric '{metric_name}' not found in runs.")

    # Filter runs that have a non-null val/loss
    filtered_runs = runs[runs[metric_name].notnull()]

    # Sort runs by val/loss ascending (lowest first)
    sorted_runs = filtered_runs.sort_values(by=metric_name, ascending=True)

    # Pick top N runs
    top_runs = sorted_runs.head(top_n)

    # Print run name (tag 'mlflow.runName' if exists), run_id, and val/loss
    print(f"Top {top_n} runs with lowest '{metric_name.replace('metrics.', '')}':")
    for _, row in top_runs.iterrows():
        run_name = row['tags.mlflow.runName'] if 'tags.mlflow.runName' in row else None
        print(f"Run Name: {run_name} | {metric_name.replace('metrics.', '')}: {row[metric_name]}")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://35.165.139.156:5000")
    experiment_name = "movenet-thunder-finetune-test"
    print_top_n_lowest_val_loss(experiment_name)