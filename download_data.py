import os
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
dataset_name = "ferraz/cotatenis-sneakers"
destination_dir = os.getcwd() + "/data"
os.makedirs(destination_dir, exist_ok=True)
api.dataset_download_files(dataset_name, path=destination_dir, unzip=True)

print("Dataset downloaded successfully.")
