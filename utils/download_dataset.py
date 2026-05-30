import os
import subprocess
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_URL = "williamscott701/memotion-dataset-7k"
POLISH_DATASET_URL = "kaszkai/polish-meme-dataset"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "memotion-dataset-7k.zip")
POLISH_ZIP_PATH = os.path.join(DATA_DIR, "polish-meme-dataset.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "memotion")
POLISH_EXTRACT_DIR = os.path.join(DATA_DIR, "polish_dataset")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACT_DIR):
        logger.info("English dataset already exists")
    else:
        logger.info("Downloading English dataset...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", DATASET_URL,
            "-p", DATA_DIR,
            "--force"
        ], check=True)

        logger.info("Extracting English dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
        os.remove(ZIP_PATH)
        logger.info("English dataset ready")

    if os.path.exists(POLISH_EXTRACT_DIR):
        logger.info("Polish dataset already exists")
    else:
        logger.info("Downloading Polish dataset...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", POLISH_DATASET_URL,
            "-p", DATA_DIR,
            "--force"
        ], check=True)

        logger.info("Extracting Polish dataset...")
        with zipfile.ZipFile(POLISH_ZIP_PATH, "r") as z:
            z.extractall(POLISH_EXTRACT_DIR)
        os.remove(POLISH_ZIP_PATH)
        logger.info("Polish dataset ready")


if __name__ == "__main__":
    download_dataset()