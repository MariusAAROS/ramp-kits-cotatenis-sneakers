import os
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
dataset_name = "ferraz/cotatenis-sneakers"
destination_dir = os.getcwd() + "/data"
os.makedirs(destination_dir, exist_ok=True)
api.dataset_download_files(dataset_name, path=destination_dir, unzip=True)

print("Dataset downloaded successfully.")


# import os
# from kaggle.api.kaggle_api_extended import KaggleApi
# import shutil
# import time
# # api = KaggleApi()
# # api.authenticate()
# # dataset_name = "ferraz/cotatenis-sneakers"
# destination_dir = os.getcwd() + "/data"
# # os.makedirs(destination_dir, exist_ok=True)
# # api.dataset_download_files(dataset_name, path=destination_dir, unzip=True)
# os.rename(destination_dir + "/sneakers_dataset", destination_dir + "/old_sneakers_dataset")
# time.sleep(0.3)
# shutil.move(destination_dir + "/old_sneakers_dataset", destination_dir)
# os.remove(destination_dir + "/old_sneakers_dataset")

# print("Dataset downloaded successfully.")