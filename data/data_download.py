import shutil
from pathlib import Path

import kagglehub

DATASET_ID = "noordeen/insurance-premium-prediction"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DATA_DIR = PROJECT_ROOT / "data"


def ensure_local_dataset() -> Path:
    print("[+] Downloading dataset from Kaggle...")

    cache_path = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"[+] Dataset downloaded to cache: {cache_path}")

    print("[+] Updating local data folder...")

    for item in PROJECT_DATA_DIR.iterdir():
        if item.name != Path(__file__).name:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    for item in cache_path.iterdir():
        destination = PROJECT_DATA_DIR / item.name

        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    print(f"[+] Data can be found here: {PROJECT_DATA_DIR}")
    print("[+] Dataset update complete.")

    return cache_path


if __name__ == "__main__":
    ensure_local_dataset()