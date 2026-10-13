import argparse
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "outputs" / "models"

MODEL_PATHS = {
    "simple": MODELS_DIR / "simple_cnn.pt",
    "complex": MODELS_DIR / "complex_cnn.pt",
    "mobilenet": MODELS_DIR / "mobilenet_v2.pt",
}


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
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

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ParallelBlock(nn.Module):
    def __init__(self, input_channels):
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

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return torch.cat([branch1, branch2, branch3], dim=1)


class ComplexCNN(nn.Module):
    def __init__(self, num_classes):
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

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.parallel_block(x)
        x = self.block3(x)
        x = self.average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.output_layer(x)


def build_model(model_name, num_classes):
    if model_name == "simple":
        model = SimpleCNN(num_classes)
    elif model_name == "complex":
        model = ComplexCNN(num_classes)
    else:
        # Không cần tải weight ImageNet vì đã có weight trong checkpoint
        model = mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model_name):
    checkpoint_path = MODEL_PATHS[model_name]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}\n"
            "Cần chạy notebook train model này trước."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        # Dành cho các bản PyTorch cũ
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    saved_model_name = checkpoint.get("model_name")
    if saved_model_name != model_name:
        raise ValueError(
            f"Checkpoint này là của model '{saved_model_name}', "
            f"không phải '{model_name}'."
        )

    class_names = checkpoint["class_names"]
    model = build_model(model_name, len(class_names))
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint không khớp với kiến trúc {model_name}: {error}"
        ) from error

    model.eval()
    return model, checkpoint


def build_eval_transform(checkpoint):
    image_size = checkpoint["image_size"]
    transform_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    normalization = checkpoint.get("normalization")
    if normalization:
        transform_steps.append(
            transforms.Normalize(
                mean=normalization["mean"],
                std=normalization["std"],
            )
        )

    return transforms.Compose(transform_steps)


def load_image(image_path, transform):
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    try:
        with Image.open(image_path) as image:
            image_tensor = transform(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"File không phải ảnh hợp lệ: {image_path}") from error

    return image_tensor.unsqueeze(0)


def predict(model_name, image_path):
    model, checkpoint = load_checkpoint(model_name)
    transform = build_eval_transform(checkpoint)
    image_tensor = load_image(image_path, transform)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probs, top_classes = torch.topk(probabilities, k=3)
    class_names = checkpoint["class_names"]

    print(f"Model: {model_name}")
    print(f"Ảnh: {image_path.expanduser().resolve()}")
    print(f"Dự đoán: {class_names[top_classes[0].item()]}")
    print(f"Xác suất: {top_probs[0].item():.2%}")
    print("Top 3 dự đoán:")

    for i, (prob, class_id) in enumerate(
        zip(top_probs.tolist(), top_classes.tolist()), start=1
    ):
        print(f"  {i}. {class_names[class_id]}: {prob:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dự đoán bệnh lá cây từ ảnh."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_PATHS.keys(),
        help="Chọn model đã train",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Đường dẫn đến ảnh cần dự đoán",
    )
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    try:
        predict(args.model, args.image)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
