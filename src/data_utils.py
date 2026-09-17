# Import thư viện
from pathlib import Path
import os
import hashlib
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imagehash
from pathlib import Path
import os
from dotenv import load_dotenv

# Load data
PROJECT_DIR = Path.cwd()
if PROJECT_DIR.name == "notebooks":
    PROJECT_DIR = PROJECT_DIR.parent

PROJECT_DIR = PROJECT_DIR.resolve()

load_dotenv(PROJECT_DIR / ".env")

DATA_DIR = Path(
    os.getenv("DATA_DIR")
    or PROJECT_DIR.parent / "PlantVillage"
).resolve()

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)

#Tìm ảnh trùng hoàn toàn
def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(8192),
            b""
        ):
            h.update(chunk)
    return h.hexdigest()

def find_exact_duplicates(df, data_dir):
    df = df.copy()

    df["sha256"] = df["relative_path"].apply(
        lambda x: get_sha256(data_dir / x)
    )

    duplicate_df = df[
        df.duplicated(
            subset="sha256",
            keep=False
        )
    ].sort_values("sha256")

    return df, duplicate_df

#Tìm ảnh gần trùng
def get_phash(path):

    try:
        with Image.open(path) as img:
            return imagehash.phash(
                img.convert("RGB")
            )

    except Exception:
        return None

def find_near_duplicates(df, data_dir, threshold=5):
    df = df.copy()

    df["phash"] = df["relative_path"].apply(
        lambda x: get_phash(data_dir / x)
    )

    near_duplicates = []

    for class_name, group in df.groupby("class_name"):

        group = group.reset_index(drop=True)

        for i in range(len(group)):
            for j in range(i + 1, len(group)):

                hash_i = group.loc[i, "phash"]
                hash_j = group.loc[j, "phash"]

                if hash_i is None or hash_j is None:
                    continue

                distance = hash_i - hash_j

                if distance <= threshold:
                    near_duplicates.append({
                        "class_name": class_name,
                        "image_1": group.loc[i, "relative_path"],
                        "image_2": group.loc[j, "relative_path"],
                        "phash_distance": distance
                    })

    return pd.DataFrame(near_duplicates)