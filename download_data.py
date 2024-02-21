import os
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import numpy as np

_download_ = True
folder = "data/sneakers_dataset/sneakers_dataset"

def load_labels(path="data/cotatenis_sneakers_kaggle.csv"):
    df = pd.read_csv(path)
    labels = df[["sku", "brand"]]
    return labels

def load_images_from_folder(folder, labels, size=999999):
    data = []
    data_labels = []
    c = 0
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
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
        except:
            print(filename)
    return data

def make_dataset(folder, build=True, size=2000):
    if build:
        labels = load_labels()
        images = load_images_from_folder(folder, labels, size=size)
        pd.DataFrame(images).to_pickle("data/sneakers_dataset.pkl")
    else:
        images = pd.read_pickle("data/sneakers_dataset.pkl")
    return images

api = KaggleApi()
api.authenticate()
dataset_name = "ferraz/cotatenis-sneakers"
destination_dir = os.getcwd() + "/data"
os.makedirs(destination_dir, exist_ok=True)

print("Downloading dataset...")
api.dataset_download_files(dataset_name, path=destination_dir, unzip=True)

print("Transforming dataset...")
cotatenis_data = make_dataset(folder, build=_download_, size=999999)
images = [img for img, _ in cotatenis_data]
labels = [brand for _, brand in cotatenis_data]
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

#save X_train, X_test as images
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)
for i, (img, label) in enumerate(zip(X_train, y_train)):
    img = Image.fromarray(img)
    img.save(f"data/train/{i}_{label}.jpg")
for i, (img, label) in enumerate(zip(X_test, y_test)):
    img = Image.fromarray(img)
    img.save(f"data/test/{i}_{label}.jpg")
pd.DataFrame(y_train).to_csv("data/train/labels.csv")
pd.DataFrame(y_test).to_csv("data/test/labels.csv")

print("Cleaning up...")
try:
    os.remove(destination_dir + "/sneakers_dataset")
    os.remove(destination_dir + "/cotatenis_sneakers_kaggle.csv")
except:
    print("Unsufficient permissions to remove files. Please remove [sneakers_dataset] and [cotatenis_sneakers_kaggle.csv] manually.")

print("Dataset downloaded successfully.")
