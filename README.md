# PlantVillage Image Classification

Project này dùng bộ dữ liệu PlantVillage để xây dựng quy trình phân loại bệnh trên lá cây bằng Deep Learning. Repo được tổ chức theo luồng làm việc thông thường của một project học máy: khám phá dữ liệu, tiền xử lý, huấn luyện mô hình, đánh giá và so sánh kết quả.

## Cấu Trúc Project

```text
proj_xu_ly_anh/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_simple_cnn.ipynb
│   ├── 04_complex_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   └── 06_comparison.ipynb
├── src/
│   ├── data_utils.py
│   ├── evaluation.py
│   └── models.py
└── outputs/
    ├── figures/
    ├── models/
    └── results/
```

## Giải Thích Thư Mục Và File

### `README.md`

File mô tả tổng quan project, cấu trúc thư mục, cách chuẩn bị dữ liệu và hướng dẫn chạy các notebook.

### `requirements.txt`

File dùng để khai báo các thư viện Python cần cài đặt cho project. Khi hoàn thiện project, các thư viện như `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow` hoặc `torch` có thể được thêm vào đây tùy theo framework sử dụng.

### `notebooks/`

Thư mục chứa các notebook theo từng bước của quy trình xây dựng mô hình.

| File                         | Mục đích                                                                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| `01_eda.ipynb`               | Khám phá dữ liệu: xem số lượng lớp, phân bố ảnh, hiển thị ảnh mẫu và nhận xét ban đầu.                |
| `02_preprocessing.ipynb`     | Làm sạch danh sách ảnh, gom nhóm ảnh trùng, chia train/validation/test và tạo pipeline ảnh.           |
| `03_simple_cnn.ipynb`        | Xây dựng và huấn luyện mô hình CNN cơ bản để làm baseline.                                            |
| `04_complex_cnn.ipynb`       | Xây dựng mô hình CNN phức tạp hơn, có thể thêm nhiều tầng convolution, dropout, batch normalization.  |
| `05_transfer_learning.ipynb` | Huấn luyện mô hình bằng transfer learning từ các mạng pretrained như MobileNet, ResNet, EfficientNet. |
| `06_comparison.ipynb`        | So sánh kết quả giữa các mô hình bằng accuracy, loss, confusion matrix và các metric đánh giá khác.   |

### `src/`

Thư mục chứa code Python dùng lại nhiều lần trong các notebook. Việc tách code vào `src/` giúp notebook gọn hơn và project dễ bảo trì hơn.

| File            | Mục đích                                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `data_utils.py` | Chứa các hàm xử lý dữ liệu: đọc ảnh, tạo dataset, resize ảnh, chuẩn hóa pixel, augmentation và chia batch. |
| `models.py`     | Chứa các hàm hoặc class định nghĩa kiến trúc mô hình: simple CNN, complex CNN, transfer learning model.    |
| `evaluation.py` | Chứa các hàm đánh giá mô hình: vẽ learning curve, confusion matrix, classification report và lưu kết quả.  |

### `outputs/`

Thư mục lưu các kết quả sinh ra trong quá trình chạy notebook.

| Thư mục            | Mục đích                                                                  |
| ------------------ | ------------------------------------------------------------------------- |
| `outputs/figures/` | Lưu biểu đồ, ảnh trực quan hóa dữ liệu, confusion matrix, learning curve. |
| `outputs/models/`  | Lưu model đã huấn luyện, checkpoint hoặc file trọng số.                   |
| `outputs/results/` | Lưu bảng kết quả, metric, file CSV hoặc báo cáo so sánh mô hình.          |

## Dữ Liệu

Dataset sử dụng:

https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Dữ liệu nên được đặt ngoài repo để tránh làm project quá nặng. Cấu trúc dữ liệu mong đợi:

```text
PlantVillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ...
└── val/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    └── ...
```

Notebook đọc `DATA_DIR` từ file `.env` ở thư mục project; nếu không có, mặc định dùng thư mục `PlantVillage` nằm cạnh project. Khi chuyển máy, cập nhật `DATA_DIR` trong `.env`.

## Cách Chạy Project

1. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

2. Mở Jupyter Notebook hoặc VS Code.

3. Chạy các notebook theo thứ tự:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_simple_cnn.ipynb
04_complex_cnn.ipynb
05_transfer_learning.ipynb
06_comparison.ipynb
```

## Kết Quả Tiền Xử Lý Hiện Tại

Notebook `02_preprocessing.ipynb` đã được chạy với dataset PlantVillage và cho kết quả:

| Split        | Số ảnh |  Tỉ lệ |
| ------------ | -----: | -----: |
| `train`      | 37,984 | 69.97% |
| `validation` |  5,427 | 10.00% |
| `test`       | 10,873 | 20.03% |

Tổng số ảnh sau khi loại 21 bản sao trùng hoàn toàn: **54,284**
Số nhãn: **38**

Quy tắc chia: toàn bộ ảnh `val` gốc và các ảnh cùng `group_id` được đưa vào
`test`. Phần `train` gốc còn lại được chia bằng `StratifiedGroupKFold` với
8 folds, stratify theo `class_name` và group theo `group_id`.

Kiểm tra split hiện tại: **0 group leakage**, **0 confirmed near-duplicate
leakage**, và cả 38 lớp đều xuất hiện trong cả ba split. Có 622 near-duplicate
ứng viên chưa được xác nhận nằm khác split; các ứng viên này không được kết luận
là leakage.

## Ghi Chú

- Không nên đưa toàn bộ dataset vào repo vì dung lượng lớn.
- Các file trong `src/` hiện là nơi để tách code dùng chung khi project được hoàn thiện.
- Các kết quả huấn luyện nên được lưu vào `outputs/` để dễ kiểm tra và so sánh.

## M3: MobileNetV2 Transfer Learning

Toàn bộ code riêng của M3 nằm trong notebook
`notebooks/05_transfer_learning.ipynb`. Notebook chỉ dùng các hàm xử lý dữ liệu
đã có trong `src/data_utils.py`; các hàm xây dựng model, huấn luyện và đánh giá
được viết ngay trong notebook để dễ theo dõi.

Mở notebook và chạy các cell theo thứ tự. Giai đoạn 1 chỉ train classifier mới.
Giai đoạn 2 nạp checkpoint tốt nhất của giai đoạn 1, mở 3 block cuối và train
với learning rate nhỏ hơn.

Checkpoint, history, cấu hình, confusion matrix và classification report được lưu
trong `outputs/models/` và `outputs/results/`. Có thể đặt `IMAGE_PATH` ở cell cuối
để dự đoán một ảnh sau khi đã train.
