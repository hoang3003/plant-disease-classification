"""Dự đoán một ảnh PlantVillage bằng checkpoint đã huấn luyện.

Ví dụ:
    python predict.py --model simple --image path/to/leaf.jpg
    python predict.py --model complex --image path/to/leaf.jpg
    python predict.py --model mobilenet --image path/to/leaf.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "outputs" / "models"
IMAGE_SIZE = 224
NUM_CLASSES = 38

CHECKPOINT_PATHS = {
    "simple": MODELS_DIR / "simple_cnn.pt",
    "complex": MODELS_DIR / "complex_cnn.pt",
    "mobilenet": MODELS_DIR / "mobilenet_v2.pt",
}

# Thứ tự này được lấy từ class_id trong preprocessing_data.csv.
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


class PredictionError(RuntimeError):
    """Lỗi dự đoán có thông báo phù hợp cho người dùng."""


def configure_console_encoding() -> None:
    """Cho phép in tiếng Việt trên terminal Windows dùng encoding cũ."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ParallelBlock(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x)],
            dim=1,
        )


class ComplexCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.parallel_block = ParallelBlock(input_channels=64)
        self.block3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.parallel_block(x)
        x = self.block3(x)
        x = self.average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.output_layer(x)


def build_model(model_name: str) -> nn.Module:
    if model_name == "simple":
        return SimpleCNN(num_classes=NUM_CLASSES)
    if model_name == "complex":
        return ComplexCNN(num_classes=NUM_CLASSES)
    if model_name == "mobilenet":
        # weights=None tránh tải ImageNet weights; checkpoint chứa toàn bộ trọng số.
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES,
        )
        return model
    raise PredictionError(f"Model không được hỗ trợ: {model_name}")


def load_checkpoint(model_name: str) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = CHECKPOINT_PATHS[model_name]
    if not checkpoint_path.is_file():
        raise PredictionError(
            f"Không tìm thấy checkpoint cho model '{model_name}': "
            f"{checkpoint_path}\n"
            "Hãy chạy notebook train tương ứng một lần để tạo checkpoint."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Tương thích với PyTorch cũ chưa có tham số weights_only.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise PredictionError(
            f"Không đọc được checkpoint {checkpoint_path}: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise PredictionError(
            f"Checkpoint {checkpoint_path} không đúng định dạng dictionary."
        )

    required_keys = {
        "format_version",
        "model_name",
        "state_dict",
        "class_names",
        "image_size",
        "normalization",
    }
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise PredictionError(
            f"Checkpoint {checkpoint_path} thiếu trường: "
            f"{', '.join(sorted(missing_keys))}."
        )

    if checkpoint["format_version"] != 1:
        raise PredictionError(
            f"Phiên bản checkpoint không hỗ trợ: "
            f"{checkpoint['format_version']}."
        )
    if checkpoint["model_name"] != model_name:
        raise PredictionError(
            f"Checkpoint không khớp model: đã chọn '{model_name}' nhưng "
            f"checkpoint là '{checkpoint['model_name']}'."
        )
    if checkpoint["class_names"] != CLASS_NAMES:
        raise PredictionError(
            "Thứ tự class trong checkpoint không khớp mapping 38 lớp "
            "của project."
        )
    if checkpoint["image_size"] != IMAGE_SIZE:
        raise PredictionError(
            f"Kích thước ảnh trong checkpoint là "
            f"{checkpoint['image_size']}, mong đợi {IMAGE_SIZE}."
        )
    if not isinstance(checkpoint["state_dict"], dict):
        raise PredictionError("state_dict trong checkpoint không hợp lệ.")

    model = build_model(model_name)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as exc:
        raise PredictionError(
            f"Checkpoint không khớp kiến trúc model '{model_name}': {exc}"
        ) from exc

    model.to("cpu")
    model.eval()
    return model, checkpoint


def build_eval_transform(
    model_name: str,
    checkpoint: dict[str, Any],
) -> transforms.Compose:
    transform_steps: list[Any] = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]

    normalization = checkpoint["normalization"]
    if model_name in {"simple", "complex"}:
        if normalization is not None:
            raise PredictionError(
                f"Checkpoint '{model_name}' chứa normalization không mong đợi."
            )
    else:
        if not isinstance(normalization, dict):
            raise PredictionError(
                "Checkpoint MobileNetV2 thiếu thông tin normalization."
            )
        mean = normalization.get("mean")
        std = normalization.get("std")
        expected = MobileNet_V2_Weights.DEFAULT.transforms()
        if list(mean or []) != list(expected.mean) or list(std or []) != list(
            expected.std
        ):
            raise PredictionError(
                "Normalization trong checkpoint MobileNetV2 không khớp "
                "transform đã dùng khi đánh giá."
            )
        transform_steps.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_steps)


def load_image(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise PredictionError(f"Không tìm thấy file ảnh: {image_path}")

    try:
        with Image.open(image_path) as image:
            image_tensor = transform(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise PredictionError(
            f"Không thể mở file như một ảnh hợp lệ: {image_path}"
        ) from exc

    return image_tensor.unsqueeze(0)


def predict(model_name: str, image_path: Path) -> None:
    model, checkpoint = load_checkpoint(model_name)
    eval_transform = build_eval_transform(model_name, checkpoint)
    image_tensor = load_image(image_path, eval_transform)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probabilities, top_indices = torch.topk(probabilities, k=3)
    predicted_index = int(top_indices[0].item())

    print(f"Model: {model_name}")
    print(f"Ảnh: {image_path.expanduser().resolve()}")
    print(f"Dự đoán: {CLASS_NAMES[predicted_index]}")
    print(f"Xác suất: {top_probabilities[0].item():.2%}")
    print("Top 3 dự đoán:")

    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities.tolist(), top_indices.tolist()),
        start=1,
    ):
        print(f"  {rank}. {CLASS_NAMES[class_index]}: {probability:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dự đoán bệnh lá cây bằng checkpoint đã train.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(CHECKPOINT_PATHS),
        help="Model cần dùng: simple, complex hoặc mobilenet.",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Đường dẫn tới ảnh lá cây cần dự đoán.",
    )
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    try:
        predict(args.model, args.image)
    except PredictionError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
