from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import numpy as np
import pickle
import os
from shutil import rmtree, copyfile

_build_ = True
_create_private_dataset_ = True
destination_dir = os.getcwd() + "/data"
folder = "data/sneakers_dataset/sneakers_dataset"

if not os.path.exists(destination_dir):
    raise FileNotFoundError(
        "The data directory does not exist. Please download it first using the ",
        "download_data.py file at the root of the directory.",
    )


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


def split_and_save(dataset, target_dir):
    if target_dir not in ["public", "private"]:
        raise ValueError("target_dir must be either 'public' or 'private'")

    os.makedirs(f"{destination_dir}/{target_dir}", exist_ok=True)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=0.3, random_state=42
    )
    os.makedirs(f"{destination_dir}/{target_dir}/train", exist_ok=True)
    os.makedirs(f"{destination_dir}/{target_dir}/test", exist_ok=True)
    for i, (path, label) in enumerate(dataset_train):
        copyfile(
            src=f"{destination_dir}/sneakers_dataset/sneakers_dataset/{path}",
            dst=f"{destination_dir}/{target_dir}/train/{path}.jpg",
        )
    for i, (path, label) in enumerate(dataset_test):
        copyfile(
            src=f"{destination_dir}/sneakers_dataset/sneakers_dataset/{path}",
            dst=f"{destination_dir}/{target_dir}/test/{path}.jpg",
        )
    pd.DataFrame(dataset_train).to_csv(
        f"{destination_dir}/{target_dir}/train/train.csv", index=False
    )
    pd.DataFrame(dataset_test).to_csv(
        f"{destination_dir}/{target_dir}/test/test.csv", index=False
    )


print("Transforming dataset...")
cotatenis_data = make_dataset(folder, build=_build_, size=999999)

print("Splitting dataset...")
if _create_private_dataset_:
    print("Splitting between public and private datasets...")
    public, private = train_test_split(cotatenis_data, test_size=0.5, random_state=42)
    split_and_save(private, "private")
    split_and_save(public, "public")
else:
    split_and_save(cotatenis_data, "public")

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
