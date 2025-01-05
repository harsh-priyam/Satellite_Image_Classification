import os
from pathlib import Path
import gdown
import yaml
import zipfile
from src.constants import CONFIG_PATH, PARAM_PATH
from src import logger


class DataIngestion:
    def __init__(self):
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)

        self.data_url = config["data_ingestion"]["data_url"]
        self.root_dir = Path(config["data_ingestion"]["root_dir"])
        self.zip_path = Path(config["data_ingestion"]["data_path"])

        os.makedirs(self.root_dir, exist_ok=True)  # Create the root directory if it doesn't exist

    def download_dataset(self):
        if self.zip_path.exists():
            # Check if the existing file is a valid ZIP file
            if not zipfile.is_zipfile(self.zip_path):
                logger.warning(f"Corrupted or invalid ZIP file found at {self.zip_path}. Deleting and re-downloading.")
                os.remove(self.zip_path)  # Delete the invalid file
            else:
                logger.info(f"File already exists at: {self.zip_path}. Skipping download.")
                return

        import re
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", self.data_url)
        if not match:
            raise ValueError(f"Invalid Google Drive URL: {self.data_url}")
        file_id = match.group(1)

        # Construct the download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"

        # Download the file
        gdown.download(download_url, str(self.zip_path), quiet=False)
        logger.info(f"Downloaded dataset to: {self.zip_path}")

    def extract_zip_dataset(self):
        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP file does not exist: {self.zip_path}")

        # Verify if the file is a valid ZIP file
        if not zipfile.is_zipfile(self.zip_path):
            logger.error(f"The file at {self.zip_path} is not a valid ZIP file.")
            raise zipfile.BadZipFile(f"The file at {self.zip_path} is not a valid ZIP file.")

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
            logger.info(f"Extracted dataset to: {self.root_dir}")


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    logger.info("------------->Data Ingestion Stage Started<-----------------")
    data_ingestion.download_dataset()
    data_ingestion.extract_zip_dataset()
