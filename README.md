# PlantVillage Image Classification

Project này dùng bộ dữ liệu PlantVillage để xây dựng quy trình phân loại bệnh trên lá cây bằng Deep Learning. Repo được tổ chức theo luồng làm việc thông thường của một project học máy: khám phá dữ liệu, tiền xử lý, huấn luyện mô hình, đánh giá và so sánh kết quả.

## Cấu Trúc Project

```text
plant-disease-classification/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_simple_cnn.ipynb
│   ├── 04_complex_cnn.ipynb       # file khung, chưa triển khai
│   └── 05_transfer_learning.ipynb
├── data/
│   ├── metadata/
│   └── processed_rgb/images/   # ảnh sinh bởi 02_preprocessing.ipynb, không commit
└── outputs/
    ├── models/
    └── results/
```

## Giải Thích Thư Mục Và File

### `README.md`

File mô tả tổng quan project, cấu trúc thư mục, cách chuẩn bị dữ liệu và hướng dẫn chạy các notebook.

### `requirements.txt`

File khai báo các thư viện Python cần cho project, bao gồm PyTorch,
torchvision, pandas, NumPy, Pillow, Matplotlib, ImageHash và scikit-learn. Cấu hình
mặc định không bắt buộc phải có GPU; các notebook model tự chọn CPU khi không
có thiết bị tăng tốc phù hợp.

### `notebooks/`

Thư mục chứa các notebook theo từng bước của quy trình xây dựng mô hình.

| File                         | Mục đích                                                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `01_eda.ipynb`               | Khám phá dữ liệu, thống kê lớp, tạo metadata và phát hiện ảnh trùng/gần trùng.          |
| `02_preprocessing.ipynb`     | Làm sạch, chia train/validation/test, resize RGB 224×224 và lưu dữ liệu đã xử lý.     |
| `03_simple_cnn.ipynb`        | Xây dựng và huấn luyện CNN cơ bản làm baseline.                                          |
| `04_complex_cnn.ipynb`       | File khung dự kiến cho CNN phức tạp; hiện chưa có nội dung triển khai.                         |
| `05_transfer_learning.ipynb` | Huấn luyện MobileNetV2 bằng transfer learning và fine-tuning, sau đó đánh giá trên tập test. |

### `outputs/`

Thư mục lưu các kết quả sinh ra trong quá trình chạy notebook.

| Thư mục            | Mục đích                                                         |
| ------------------ | ---------------------------------------------------------------- |
| `outputs/models/`  | Lưu checkpoint hoặc file trọng số sinh ra khi huấn luyện.       |
| `outputs/results/` | Lưu manifest, cấu hình, lịch sử train và báo cáo đánh giá. |

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

Notebook đọc `DATA_DIR` từ file `.env` ở thư mục project. Ví dụ:

```dotenv
DATA_DIR=C:/datasets/PlantVillage
```

Nếu không có `.env`, notebook mặc định tìm thư mục `PlantVillage` nằm cạnh
project. File `.env` chỉ chứa cấu hình riêng của từng máy và không được commit.

Sau khi chạy `02_preprocessing.ipynb`, ảnh sạch được chuyển sang RGB, resize `224×224`
và lưu trong một thư mục chung. Thông tin nhãn và split nằm trong CSV:

```text
data/processed_rgb/
└── images/
    ├── <image_1>.jpg
    └── ...

outputs/results/
└── processed_data.csv
```

Mỗi dòng `processed_data.csv` chứa `processed_path`, `class_name`, `class_id`, `split`
và các cột truy vết từ `data_split.csv`. Thư mục ảnh được bỏ qua bởi Git; file CSV
được giữ trong project. Augmentation không được ghi cố định vào validation/test;
notebook train tạo augmentation ngẫu nhiên cho train và thực hiện normalization phù
hợp với từng model.

## Cách Chạy Project

1. Tạo và kích hoạt môi trường Python riêng cho project (khuyến nghị):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

3. Tạo file `.env` và khai báo `DATA_DIR` nếu dataset không nằm cạnh project.

4. Mở Jupyter Notebook hoặc VS Code, sau đó chạy:

```text
01_eda.ipynb
02_preprocessing.ipynb
├── 03_simple_cnn.ipynb          # CNN baseline
└── 05_transfer_learning.ipynb  # MobileNetV2
```

Hai notebook model có thể chạy độc lập sau khi preprocessing hoàn tất. Notebook
`04_complex_cnn.ipynb` chưa triển khai nên không nằm trong quy trình chạy hiện tại.

## Kết Quả Tiền Xử Lý Hiện Tại

Notebook `02_preprocessing.ipynb` tạo manifest và bộ ảnh RGB `224×224` với kết quả:

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
- `01_eda.ipynb` và `02_preprocessing.ipynb` tự chứa các hàm xử lý dữ liệu cần thiết.
- Chỉ lưu kết quả huấn luyện vào `outputs/` khi thực sự cần dùng lại.

## M3: MobileNetV2 Transfer Learning

Toàn bộ code riêng của M3 nằm trong notebook
`notebooks/05_transfer_learning.ipynb`. Notebook đọc `preprocessing_data.csv`, lọc
theo cột `split` và mở ảnh gốc qua `relative_path`. Các hàm xây dựng model, huấn
luyện và đánh giá được viết ngay trong notebook để dễ theo dõi.

Mở notebook và chạy các cell theo thứ tự. Giai đoạn 1 chỉ train classifier mới.
Giai đoạn 2 dùng lại trọng số tốt nhất đang giữ trong RAM, mở 3 block cuối và
train với learning rate nhỏ hơn.

Notebook hiển thị history, confusion matrix và classification report trực tiếp,
không tự động ghi checkpoint hay báo cáo ra file.
