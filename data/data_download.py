import shutil
from pathlib import Path

import kagglehub

DATASET_ID = "noordeen/insurance-premium-prediction"
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DATA_DIR = PROJECT_ROOT / "data"

def ensure_local_dataset() -> Path:
    cache_path = Path(kagglehub.dataset_download(DATASET_ID))

    if PROJECT_DATA_DIR.exists():
        shutil.rmtree(PROJECT_DATA_DIR)
	
    shutil.copytree(cache_path, PROJECT_DATA_DIR)
    return cache_path

if __name__ == "__main__":
    ensure_local_dataset()