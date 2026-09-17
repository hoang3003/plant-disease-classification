# Import thư viện
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import os
import hashlib
import json

import pandas as pd
from PIL import Image
import imagehash
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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

    with ThreadPoolExecutor(max_workers=16) as pool:
        df["sha256"] = list(pool.map(
            get_sha256,
            (_image_path(data_dir, path) for path in df["relative_path"])
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

    with ThreadPoolExecutor(max_workers=16) as pool:
        df["phash"] = [
            str(value) if value is not None else None
            for value in pool.map(
                get_phash,
                (_image_path(data_dir, path) for path in df["relative_path"])
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


# Các hàm dùng cho 02_preprocessing.ipynb.
SPLIT_COLUMNS = [
    "relative_path", "class_name", "class_id", "split", "original_split", "group_id"
]


def normalize_relpath(value):
    """Chuẩn hóa đường dẫn tương đối để CSV dùng được trên nhiều hệ điều hành."""
    value = str(value).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _normalize_path_column(dataframe, column):
    dataframe[column] = dataframe[column].astype(str).map(normalize_relpath)


def _image_path(data_dir, relative_path):
    return Path(data_dir).joinpath(*PurePosixPath(normalize_relpath(relative_path)).parts)


def _read_csv_or_empty(path, expected_columns):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=expected_columns)


def _require_columns(dataframe, columns, filename):
    missing = set(columns) - set(dataframe.columns)
    if missing:
        raise ValueError(f"{filename} thiếu các cột: {sorted(missing)}")


def load_preprocessing_inputs(metadata_path, duplicate_path, near_duplicate_path,
                              broken_path):
    """Đọc các CSV từ EDA, kiểm tra cột và chuẩn hóa đường dẫn ảnh."""
    metadata = pd.read_csv(metadata_path).copy()
    _require_columns(metadata, ("relative_path", "split", "class_name"), "metadata.csv")
    _normalize_path_column(metadata, "relative_path")
    metadata["original_split"] = metadata["split"].astype(str).str.lower()
    metadata["class_name"] = metadata["class_name"].astype(str)

    duplicate_df = _read_csv_or_empty(
        duplicate_path, ["relative_path", "split", "class_name", "sha256"]
    )
    _require_columns(duplicate_df, ("relative_path", "sha256"), "duplicate_images.csv")
    _normalize_path_column(duplicate_df, "relative_path")

    near_duplicate_df = _read_csv_or_empty(
        near_duplicate_path, ["class_name", "image_1", "image_2", "phash_distance"]
    )
    _require_columns(near_duplicate_df, ("image_1", "image_2"), "near_duplicates.csv")
    for column in ("image_1", "image_2"):
        _normalize_path_column(near_duplicate_df, column)

    if Path(broken_path).exists():
        broken_df = _read_csv_or_empty(broken_path, ["relative_path", "error"])
        _require_columns(broken_df, ("relative_path", "error"), "broken_images.csv")
        _normalize_path_column(broken_df, "relative_path")
    else:
        broken_df = pd.DataFrame(columns=["relative_path", "error"])

    return metadata, duplicate_df, near_duplicate_df, broken_df


def find_broken_and_missing_images(metadata, data_dir, broken_df=None,
                                   verify_images=False, max_workers=16):
    """Tìm file thiếu; tùy chọn kiểm tra ảnh lỗi mà không giữ ảnh trong RAM."""
    if broken_df is None:
        broken_df = pd.DataFrame(columns=["relative_path", "error"])
    broken_paths = set(
        broken_df.get("relative_path", pd.Series(dtype=str)).dropna().astype(str)
    )
    missing_paths = {
        path for path in metadata["relative_path"]
        if not _image_path(data_dir, path).exists()
    }

    if verify_images:
        def verify_one(relative_path):
            try:
                with Image.open(_image_path(data_dir, relative_path)) as image:
                    image.verify()
                return None
            except Exception as error:
                return {"relative_path": relative_path, "error": str(error)}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            records = [
                result for result in pool.map(verify_one, metadata["relative_path"])
                if result is not None
            ]
        checked_df = pd.DataFrame(records, columns=["relative_path", "error"])
        broken_df = pd.concat([broken_df, checked_df], ignore_index=True)
        broken_paths.update(checked_df["relative_path"])

    return broken_df, broken_paths, missing_paths


def build_duplicate_groups(metadata, duplicate_df, near_duplicate_df):
    """Gom ảnh trùng/gần trùng và phát hiện nhóm chứa nhiều nhãn."""
    data = metadata.copy()
    paths = data["relative_path"].tolist()
    if len(paths) != len(set(paths)):
        raise ValueError("metadata.csv có relative_path bị lặp.")
    known_paths = set(paths)
    parent = {path: path for path in paths}
    rank = {path: 0 for path in paths}

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(first, second):
        if first not in parent or second not in parent:
            return
        root_a, root_b = find(first), find(second)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    if not duplicate_df.empty:
        _require_columns(duplicate_df, ("relative_path", "sha256"), "duplicate_images.csv")
        for _, group in duplicate_df.groupby("sha256"):
            members = [
                path for path in group["relative_path"].dropna().astype(str)
                if path in known_paths
            ]
            for path in members[1:]:
                union(members[0], path)

    if not near_duplicate_df.empty:
        _require_columns(near_duplicate_df, ("image_1", "image_2"), "near_duplicates.csv")
        for row in near_duplicate_df.itertuples(index=False):
            union(normalize_relpath(row.image_1), normalize_relpath(row.image_2))

    components = {}
    for path in paths:
        components.setdefault(find(path), []).append(path)
    path_to_group = {
        path: f"group_{index:05d}"
        for index, members in enumerate(
            sorted(components.values(), key=lambda members: min(members))
        )
        for path in members
    }
    data["group_id"] = data["relative_path"].map(path_to_group)
    label_counts = data.groupby("group_id")["class_name"].nunique()
    conflict_groups = set(label_counts[label_counts > 1].index)
    conflict_df = data[data["group_id"].isin(conflict_groups)].sort_values(
        ["group_id", "class_name", "relative_path"]
    )
    return data, conflict_groups, conflict_df


def clean_metadata(metadata, duplicate_df, broken_paths, missing_paths,
                   label_conflict_df):
    """Loại ảnh lỗi, nhóm sai nhãn và bản sao SHA256; ưu tiên ảnh val."""
    excluded_reasons = {}

    def exclude(path, reason):
        excluded_reasons.setdefault(normalize_relpath(path), set()).add(reason)

    for path in broken_paths:
        exclude(path, "broken_image")
    for path in missing_paths:
        exclude(path, "missing_file")
    conflict_paths = set(label_conflict_df["relative_path"])
    for path in conflict_paths:
        exclude(path, "label_conflict_group")

    exact_duplicate_removed = set()
    if not duplicate_df.empty:
        _require_columns(duplicate_df, ("relative_path", "sha256"), "duplicate_images.csv")
        lookup = metadata.set_index("relative_path")[
            ["original_split", "class_name", "group_id"]
        ]
        invalid_paths = set(broken_paths) | set(missing_paths) | conflict_paths
        for _, group in duplicate_df.groupby("sha256"):
            candidates = sorted({
                path for path in group["relative_path"].dropna().astype(str)
                if path in lookup.index and path not in invalid_paths
            })
            if len(candidates) <= 1:
                continue
            val_candidates = [
                path for path in candidates
                if lookup.loc[path, "original_split"] == "val"
            ]
            keep_path = val_candidates[0] if val_candidates else candidates[0]
            for path in candidates:
                if path != keep_path:
                    exact_duplicate_removed.add(path)
                    exclude(path, "exact_duplicate_copy")

    data = metadata.copy()
    data["excluded"] = data["relative_path"].isin(excluded_reasons)
    clean_df = data[~data["excluded"]].copy()
    return clean_df, excluded_reasons, exact_duplicate_removed


def build_class_mapping(metadata, clean_df):
    """Tạo thứ tự class cố định và gán class_id cho ảnh đã làm sạch."""
    class_names = sorted(metadata["class_name"].dropna().unique().tolist())
    missing_classes = set(class_names) - set(clean_df["class_name"].unique())
    if missing_classes:
        raise ValueError(
            f"Sau làm sạch có class bị mất hoàn toàn: {sorted(missing_classes)}"
        )
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    clean_df = clean_df.copy()
    clean_df["class_id"] = clean_df["class_name"].map(class_to_idx).astype(int)
    return clean_df, class_names, class_to_idx, idx_to_class


def build_data_split(clean_df, seed=42, n_splits=8):
    """Đưa group chạm val gốc vào test, chia phần train còn lại theo group."""
    data = clean_df.copy()
    test_group_ids = set(data.loc[data["original_split"] == "val", "group_id"])
    data["split"] = None
    data.loc[data["group_id"].isin(test_group_ids), "split"] = "test"
    train_pool = data[data["split"].isna()].copy()
    if (train_pool["original_split"] != "train").any():
        raise ValueError("Có ảnh chưa gán split nhưng không thuộc train gốc.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    candidates = []
    target_ratio = 1 / n_splits
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(
        train_pool, y=train_pool["class_name"], groups=train_pool["group_id"]
    )):
        fold_val = train_pool.iloc[val_idx]
        overall_error = abs(len(fold_val) / len(train_pool) - target_ratio)
        per_class_ratio = (
            fold_val["class_name"].value_counts()
            / train_pool["class_name"].value_counts()
        ).fillna(0)
        class_error = per_class_ratio.sub(target_ratio).abs().mean()
        candidates.append((
            float(overall_error + class_error), fold_id, train_idx, val_idx
        ))

    _, selected_fold, train_idx, val_idx = min(
        candidates, key=lambda item: item[0]
    )
    data.loc[train_pool.index[train_idx], "split"] = "train"
    data.loc[train_pool.index[val_idx], "split"] = "validation"
    if data["split"].isna().any():
        raise RuntimeError("Có ảnh chưa được gán train/validation/test.")
    return data, selected_fold, len(test_group_ids)


def load_or_create_split(clean_df, path, force_rebuild=False, seed=42, n_splits=8,
                         config_path=None):
    """Dùng lại split khớp metadata hoặc tạo split mới; chưa ghi file."""
    path = Path(path)
    if path.exists() and not force_rebuild:
        saved_split = pd.read_csv(path)
        _require_columns(saved_split, SPLIT_COLUMNS, "data_split.csv")
        _normalize_path_column(saved_split, "relative_path")
        identity_columns = [
            "relative_path", "class_name", "class_id", "original_split", "group_id"
        ]
        expected = clean_df[identity_columns].sort_values(
            "relative_path"
        ).reset_index(drop=True)
        actual = saved_split[identity_columns].sort_values(
            "relative_path"
        ).reset_index(drop=True)
        if not saved_split["relative_path"].is_unique or not actual.equals(expected):
            raise ValueError(
                "data_split.csv không còn khớp metadata/nhóm ảnh hiện tại."
            )
        test_group_count = saved_split.loc[
            saved_split["split"] == "test", "group_id"
        ].nunique()
        selected_fold = None
        if config_path is not None and Path(config_path).exists():
            with Path(config_path).open("r", encoding="utf-8") as file:
                previous_config = json.load(file)
            recorded_fold = previous_config.get("split_strategy", {}).get("selected_fold")
            if type(recorded_fold) is int and 0 <= recorded_fold < n_splits:
                selected_fold = recorded_fold

        # Config cũ có thể đã ghi null khi dùng lại split. Chỉ khôi phục fold
        # nếu cách chia hiện tại tạo đúng cùng tập ảnh đã lưu.
        if selected_fold is None:
            recreated, inferred_fold, _ = build_data_split(
                clean_df, seed=seed, n_splits=n_splits
            )
            saved_assignment = saved_split.set_index("relative_path")["split"].sort_index()
            new_assignment = recreated.set_index("relative_path")["split"].sort_index()
            if (saved_assignment.index.equals(new_assignment.index)
                    and (saved_assignment.to_numpy() == new_assignment.to_numpy()).all()):
                selected_fold = inferred_fold
        return saved_split, selected_fold, test_group_count, True

    split_df, selected_fold, test_group_count = build_data_split(
        clean_df, seed=seed, n_splits=n_splits
    )
    split_df = split_df[SPLIT_COLUMNS].sort_values(
        ["split", "class_id", "relative_path"]
    ).reset_index(drop=True)
    return split_df, selected_fold, test_group_count, False


def validate_split(split_df, near_duplicate_df, class_names):
    """Kiểm tra path, group, val gốc và cặp ảnh gần trùng giữa các tập."""
    _require_columns(split_df, SPLIT_COLUMNS, "data_split.csv")
    if not split_df["relative_path"].is_unique:
        raise ValueError("Có relative_path bị lặp trong data_split.csv.")
    if split_df["split"].isna().any() or not set(split_df["split"]).issubset(
        {"train", "validation", "test"}
    ):
        raise ValueError("Có giá trị split không hợp lệ.")

    group_overlap = split_df.groupby("group_id")["split"].nunique()
    leaking_groups = group_overlap[group_overlap > 1]
    if not leaking_groups.empty:
        raise ValueError(
            f"Có {len(leaking_groups)} group bị leakage giữa các split."
        )
    original_val_not_test = split_df[
        (split_df["original_split"] == "val")
        & (split_df["split"] != "test")
    ]
    if not original_val_not_test.empty:
        raise ValueError("Có ảnh val gốc không nằm trong test.")

    path_to_split = dict(zip(split_df["relative_path"], split_df["split"]))
    near_pair_leaks = []
    if not near_duplicate_df.empty:
        _require_columns(near_duplicate_df, ("image_1", "image_2"), "near_duplicates.csv")
        for row in near_duplicate_df.itertuples(index=False):
            first = normalize_relpath(row.image_1)
            second = normalize_relpath(row.image_2)
            if (first in path_to_split and second in path_to_split
                    and path_to_split[first] != path_to_split[second]):
                near_pair_leaks.append((first, second))
    if near_pair_leaks:
        raise ValueError(
            f"Có {len(near_pair_leaks)} near-duplicate pair bị tách split."
        )

    counts = split_df["split"].value_counts().reindex(
        ["train", "validation", "test"], fill_value=0
    )
    split_summary = pd.DataFrame({
        "count": counts,
        "ratio": (counts / len(split_df)).round(4),
    })
    class_split_counts = split_df.pivot_table(
        index=["class_id", "class_name"], columns="split",
        values="relative_path", aggfunc="count", fill_value=0
    ).reindex(columns=["train", "validation", "test"], fill_value=0)
    class_split_counts["total"] = class_split_counts.sum(axis=1)
    missing_class_rows = []
    for split_name in ("train", "validation", "test"):
        class_counts = split_df.loc[
            split_df["split"] == split_name, "class_name"
        ].value_counts().reindex(class_names, fill_value=0)
        missing_classes = class_counts[class_counts == 0].index.tolist()
        if missing_classes:
            missing_class_rows.append({
                "split": split_name,
                "missing_classes": missing_classes,
            })
    return {
        "leaking_groups": leaking_groups,
        "near_pair_leaks": near_pair_leaks,
        "split_summary": split_summary,
        "class_split_counts": class_split_counts,
        "missing_class_rows": missing_class_rows,
    }


def make_transforms(image_size=224):
    """Resize mọi ảnh; augmentation chỉ áp dụng cho train."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, scale=(0.90, 1.10)),
        transforms.ColorJitter(contrast=0.10),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return train_transform, eval_transform


class PlantVillageSplitDataset(Dataset):
    """Chỉ mở ảnh khi DataLoader yêu cầu một phần tử."""

    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        with Image.open(_image_path(self.data_dir, row["relative_path"])) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["class_id"])


def make_dataloaders(split_df, data_dir, train_transform, eval_transform,
                     batch_size=32, num_workers=0, pin_memory=None):
    """Tạo DataLoader cho train, validation và test theo thứ tự đó."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loaders = []
    for split_name in ("train", "validation", "test"):
        subset = split_df[split_df["split"] == split_name]
        transform = train_transform if split_name == "train" else eval_transform
        dataset = PlantVillageSplitDataset(subset, data_dir, transform=transform)
        loaders.append(DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        ))
    return tuple(loaders)


def save_preprocessing_outputs(split_df, class_names, config, split_path,
                               class_names_path, config_path, save_split=True):
    """Lưu split đã kiểm tra, thứ tự class và cấu hình preprocessing."""
    split_path = Path(split_path)
    class_names_path = Path(class_names_path)
    config_path = Path(config_path)
    for path in (split_path, class_names_path, config_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if save_split:
        split_df.to_csv(split_path, index=False)
    with class_names_path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
