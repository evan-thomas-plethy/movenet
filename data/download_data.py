import subprocess
import os
import re

run_ids = [
    "d04e88501d344f0c983feecbf5bd4b94"
]
local_base_dir = "dataset_3.0"

bucket = "visionai1"
experiment_id = 3
base_s3_prefix = f"s3://{bucket}/mlflow-artifacts/{experiment_id}"

os.makedirs(local_base_dir, exist_ok=True)

version_pattern = re.compile(r'v(\d+)/?$')

for run_id in run_ids:
    dataset_prefix = f"{base_s3_prefix}/{run_id}/artifacts/dataset/"
    print(f"Listing versions under {dataset_prefix}")

    # List objects/folders under dataset prefix
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", dataset_prefix],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list versions for run_id: {run_id}")
        print(e)
        continue

    # Parse the output lines for folders named vX/
    versions = []
    for line in result.stdout.splitlines():
        # Example line format for folders: "                           PRE v1/"
        parts = line.strip().split()
        if len(parts) == 2 and parts[0] == "PRE":
            folder_name = parts[1]
            match = version_pattern.match(folder_name)
            if match:
                versions.append(int(match.group(1)))

    if not versions:
        print(f"⚠️ No version folders found under {dataset_prefix} for run_id {run_id}")
        continue

    max_version = max(versions)
    max_version_folder = f"v{max_version}"
    s3_path = f"{dataset_prefix}{max_version_folder}/"

    print(f"Downloading latest version {max_version_folder} from {s3_path} to {local_base_dir}/ ...")

    try:
        subprocess.run([
            "aws", "s3", "cp", s3_path, local_base_dir, "--recursive"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download for run_id: {run_id}")
        print(e)