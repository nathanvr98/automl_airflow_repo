import json
from pendulum import datetime
import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import logging
from airflow.decorators import (
    dag,
    task,
)

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Fill the .env file with your Kaggle API keys

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

if not KAGGLE_USERNAME or not KAGGLE_KEY:
    logging.error("KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment.")
else:
    logging.info("Kaggle credentials loaded successfully.")


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 2,
    },
    tags=["pycaret"],
)
def pycaret_dag():
    @task(task_id="GetDataKaggle")
    def extract_data_from_kaggle():

        competition_name = "playground-series-s4e6"
        download_path = "data/"

        if not os.path.exists(download_path):
            os.makedirs(download_path)
            logging.info(f"Created directory: {download_path}")

        api = KaggleApi()
        api.authenticate()
        logging.info("Kaggle API authenticated successfully.")

        api.competition_download_files(competition_name, path=download_path)
        logging.info(f"Files downloaded to {download_path}")

        # Unzip the downloaded files
        for item in os.listdir(download_path):
            if item.endswith(".zip"):
                zip_ref = zipfile.ZipFile(os.path.join(download_path, item), "r")
                zip_ref.extractall(download_path)
                zip_ref.close()
                logging.info(f"Unzipped {item}")

                extracted_files = zip_ref.namelist()
                logging.info(f"Extracted files: {extracted_files}")

        # List all files in the directory after extraction
        for root, dirs, files in os.walk(download_path):
            for name in files:
                logging.info(f"File found: {os.path.join(root, name)}")

    @task(task_id="AutoML_PyCaret")
    def auto_ml_pycaret():
        return logging.info("AutoML solution created")

    @task(task_id="SubmitKaggle")
    def submit_to_kaggle():
        api = KaggleApi()
        api.authenticate()
        logging.info("Kaggle API authenticated successfully.")
        api.competition_submit(
            file_name="submissions/submission_0.csv",
            message="First submission",
            competition="playground-series-s4e4",
        )
        return logging.info("Solution submitted to Kaggle")

    extract_data_from_kaggle()


dag = pycaret_dag()
