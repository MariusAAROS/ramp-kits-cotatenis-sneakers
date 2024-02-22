import os
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import numpy as np
import pickle
from shutil import rmtree, copyfile

_download_ = True
folder = "data/sneakers_dataset/sneakers_dataset"


def load_labels(path="data/cotatenis_sneakers_kaggle.csv"):
    df = pd.read_csv(path)
    labels = df[["sku", "brand"]]
    return labels


def load_images_from_folder(folder, labels, size):
    data = []
    # data_labels = []
    c = 0
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        np_img = np.array(img)
        img.close()
        try:
            sku = filename.split("_")[0]
            brand = labels[labels["sku"] == sku]["brand"].values[0]
            if img is not None:
                data.append((np_img, brand))
            c += 1
            if c >= size:
                break
        except Exception:
            print(f"Brand not found for file {filename}. This is normal and expected.")
    return data


def load_image_paths_from_folder(folder, labels, size):
    data = []
    c = 0
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img.close()
        try:
            sku = filename.split("_")[0]
            brand = labels[labels["sku"] == sku]["brand"].values[0]
            if img is not None:
                data.append((filename, brand))
            c += 1
            if c >= size:
                break
        except Exception:
            print(f"Brand not found for file {filename}. This is normal and expected.")
    return data


def make_dataset(folder, build, size):
    if build:
        labels = load_labels()
        dataset = load_image_paths_from_folder(folder, labels, size=size)
        with open("data/sneakers_dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)
    else:
        with open("data/sneakers_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
    return dataset


api = KaggleApi()
api.authenticate()
dataset_name = "ferraz/cotatenis-sneakers"
destination_dir = os.getcwd() + "/data"
os.makedirs(destination_dir, exist_ok=True)


print("Downloading dataset...")
api.dataset_download_files(dataset_name, path=destination_dir, unzip=True)

print("Transforming dataset...")
cotatenis_data = make_dataset(folder, build=_download_, size=999999)
paths = [path for path, _ in cotatenis_data]
labels = [brand for _, brand in cotatenis_data]
X_train, X_test, y_train, y_test = train_test_split(
    paths, labels, test_size=0.3, random_state=42
)

print("Splitting dataset...")
# save X_train, X_test as images
os.makedirs(f"{destination_dir}/train", exist_ok=True)
os.makedirs(f"{destination_dir}/test", exist_ok=True)
for i, (path, label) in enumerate(zip(X_train, y_train)):
    copyfile(
        src=f"{destination_dir}/sneakers_dataset/sneakers_dataset/{path}",
        dst=f"{destination_dir}/train/{i}.jpg",
    )
for i, (img, label) in enumerate(zip(X_test, y_test)):
    copyfile(
        src=f"{destination_dir}/sneakers_dataset/sneakers_dataset/{path}",
        dst=f"{destination_dir}/test/{i}.jpg",
    )
pd.DataFrame(y_train).to_csv(f"{destination_dir}/train/labels.csv", index=False)
pd.DataFrame(y_test).to_csv(f"{destination_dir}/test/labels.csv", index=False)

print("Cleaning up...")
try:
    rmtree(
        destination_dir + "/sneakers_dataset"
    )  # faire attention à cette commande, elle détruit tout sur son passage
    os.remove(destination_dir + "/cotatenis_sneakers_kaggle.csv")
except Exception as e:
    raise PermissionError(
        "Unsufficient permissions to remove files.",
        "Please remove [sneakers_dataset] and [cotatenis_sneakers_kaggle.csv]",
        " manually.",
        f"Error message: {e}",
    )

print("Dataset downloaded successfully.")
