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

# Tìm ảnh trùng hoàn toàn
import hashlib

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(8192),
            b""
        ):
            h.update(chunk)
    return h.hexdigest()

df["sha256"] = df["relative_path"].apply(
    lambda x: get_sha256(DATA_DIR / x)
)

duplicate_df = df[
    df.duplicated(
        subset="sha256",
        keep=False
    )
].sort_values("sha256")

duplicate_df[
    ["relative_path", "split", "class_name", "sha256"]
]

# Kiểm tra trùng hoàn toàn giữa train và val
duplicate_cross_split = (
    duplicate_df
    .groupby("sha256")["split"]
    .nunique()
)

cross_split_hashes = duplicate_cross_split[
    duplicate_cross_split > 1
].index

cross_split_duplicates = duplicate_df[
    duplicate_df["sha256"].isin(
        cross_split_hashes
    )
]

cross_split_duplicates[
    ["relative_path", "split", "class_name", "sha256"]
]

# Tìm ảnh gần trùng
def get_phash(path):

    try:
        with Image.open(path) as img:
            return imagehash.phash(
                img.convert("RGB")
            )

    except Exception:
        return None

df["phash"] = df["relative_path"].apply(
    lambda x: str(get_phash(DATA_DIR / x))
)

near_duplicates = []

for class_name, group in df.groupby("class_name"):

    group = group.reset_index(drop=True)

    hashes = [
        imagehash.hex_to_hash(x)
        for x in group["phash"]
        if pd.notna(x)
    ]

    paths = group["relative_path"].tolist()

    for i in range(len(hashes)):

        for j in range(i + 1, len(hashes)):

            distance = hashes[i] - hashes[j]

            if distance <= 5:

                near_duplicates.append({
                    "class_name": class_name,
                    "image_1": paths[i],
                    "image_2": paths[j],
                    "phash_distance": distance
                })

near_duplicate_df = pd.DataFrame(
    near_duplicates
)
near_duplicate_df.head()