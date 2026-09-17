# Import thư viện
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import hashlib

import pandas as pd
from PIL import Image
import imagehash
from dotenv import load_dotenv

# Load data
PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_DIR / "data" / "metadata"

load_dotenv(PROJECT_DIR / ".env")

data_dir_value = os.getenv("DATA_DIR")
DATA_DIR = Path(data_dir_value) if data_dir_value else PROJECT_DIR.parent / "PlantVillage"
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_DIR / DATA_DIR
DATA_DIR = DATA_DIR.resolve()

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
    data_dir = Path(data_dir)

    with ThreadPoolExecutor(max_workers=16) as pool:
        df["sha256"] = list(pool.map(
            get_sha256,
            (data_dir / path for path in df["relative_path"])
        ))

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
    data_dir = Path(data_dir)

    with ThreadPoolExecutor(max_workers=16) as pool:
        df["phash"] = [
            str(value) if value is not None else None
            for value in pool.map(
                get_phash,
                (data_dir / path for path in df["relative_path"])
            )
        ]

    near_duplicates = []

    for class_name, group in df.groupby("class_name"):
        valid_rows = group[group["phash"].notna()]
        hashes = [int(value, 16) for value in valid_rows["phash"]]
        paths = valid_rows["relative_path"].tolist()

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                distance = (hashes[i] ^ hashes[j]).bit_count()
                if distance <= threshold:
                    near_duplicates.append({
                        "class_name": class_name,
                        "image_1": paths[i],
                        "image_2": paths[j],
                        "phash_distance": distance
                    })

    return pd.DataFrame(near_duplicates)
