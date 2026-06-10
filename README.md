# Phân loại ung thư phổi LUAD/LUSC bằng MIL kết hợp nới lỏng Lagrange và trọng số mờ bất đối xứng

## Giới thiệu

Dự án xây dựng mô hình phân loại hai tiểu loại ung thư phổi không tế bào nhỏ gồm:

- Ung thư biểu mô tuyến phổi (LUAD)
- Ung thư biểu mô tế bào vảy phổi (LUSC)

trên bộ dữ liệu mô bệnh học số hóa TCGA-Lung.

Do ảnh mô bệnh học toàn tiêu bản (Whole Slide Image - WSI) có kích thước rất lớn và chỉ được gán nhãn ở cấp độ tiêu bản, bài toán được mô hình hóa dưới dạng **Multiple Instance Learning (MIL)**, trong đó:

- Mỗi tiêu bản WSI được xem là một **túi (bag)**.
- Mỗi mảnh ảnh (**patch**) được xem là một **đối tượng (instance)**.
- Nhãn chỉ được cung cấp ở cấp độ túi.

Mô hình đề xuất kết hợp:

- Multiple Instance Learning (MIL)
- Nới lỏng Lagrange (Lagrangian Relaxation)
- Trọng số mờ bất đối xứng (Asymmetric Fuzzy Weighting)
- Máy vector hỗ trợ tuyến tính (Linear SVM)
- Giảm chiều dữ liệu bằng Incremental PCA

nhằm nâng cao hiệu quả phân loại trên dữ liệu mô bệnh học có kích thước lớn.

---

## Kiến trúc phương pháp

### Bước 1. Trích xuất đặc trưng

Các ảnh WSI được chia thành nhiều patch nhỏ.

Đặc trưng của từng patch được trích xuất bằng mạng CNN ResNet-50 và lưu dưới dạng các tệp `.pt`.

---

### Bước 2. Chuẩn hóa và giảm chiều

Đặc trưng được:

- Chuẩn hóa bằng `StandardScaler`
- Giảm chiều bằng `Incremental PCA`

Số chiều sau giảm:

```
2048 → 400 chiều
```

Việc sử dụng Incremental PCA giúp xử lý hiệu quả tập dữ liệu lớn mà không yêu cầu toàn bộ dữ liệu phải được nạp vào bộ nhớ cùng lúc.

---

### Bước 3. MIL kết hợp Nới lỏng Lagrange

Mỗi WSI được xem như một túi gồm nhiều instance.

Các ràng buộc MIL được đưa vào hàm mục tiêu thông qua các nhân tử Lagrange để xây dựng bài toán đối ngẫu.

Quá trình tối ưu được thực hiện theo chiến lược lặp:

1. Cố định nhãn instance và huấn luyện SVM.
2. Cập nhật nhãn instance.
3. Cập nhật nhân tử Lagrange bằng thuật toán dưới đạo hàm (Sub-gradient).
4. Lặp lại cho đến khi hội tụ.

---

### Bước 4. Trọng số mờ bất đối xứng

Đối với:

- Túi LUSC: tất cả instance được gán trọng số bằng 1.
- Túi LUAD: sử dụng cơ chế fuzzy để giảm ảnh hưởng của các instance nhiễu.

Cách tiếp cận này giúp mô hình thích ứng tốt hơn với đặc tính không đồng nhất của dữ liệu mô bệnh học.

---

## Cấu trúc dữ liệu

### File nhãn

```csv
labels.csv

wsi_id,label
TCGA-XX-XXXX,LUAD
TCGA-XX-XXXX,LUSC
...
```

### Thư mục đặc trưng

```text
pt_files/
├── TCGA-01.pt
├── TCGA-02.pt
├── TCGA-03.pt
└── ...
```

Mỗi tệp `.pt` chứa đặc trưng ResNet-50 của toàn bộ các patch thuộc một WSI.

---

## Bộ dữ liệu

Bộ dữ liệu sử dụng trong nghiên cứu là:

**TCGA-Lung**

gồm hai nhóm bệnh:

- Lung Adenocarcinoma (LUAD)
- Lung Squamous Cell Carcinoma (LUSC)

Dữ liệu đặc trưng được tải từ:

https://zenodo.org/records/10563985

Bộ dữ liệu bao gồm các vector đặc trưng đã được trích xuất từ các ảnh mô bệnh học thuộc dự án The Cancer Genome Atlas (TCGA), phục vụ cho nghiên cứu về Multiple Instance Learning trong phân loại ung thư.

---

## Yêu cầu môi trường

- Python 3.10 trở lên

### Cài đặt thư viện

```bash
pip install numpy
pip install pandas
pip install torch
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

Hoặc:

```bash
pip install -r requirements.txt
```

---

## Cấu hình đường dẫn dữ liệu

Trong file mã nguồn, chỉnh sửa:

```python
DUONG_DAN_CSV = "D:/LUNG/labels.csv"
DUONG_DAN_THU_MUC_PT = "D:/LUNG/LUNG/pt_files"
```

thành đường dẫn tương ứng trên máy tính của bạn.

---

## Cách chạy chương trình

Thực hiện lệnh:

```bash
python AsymFuzzyOpt.py
```

---

## Quy trình đánh giá

Mô hình được đánh giá bằng:

- Stratified 5-Fold Cross Validation
- Grid Search lựa chọn tham số phạt C
- ROC Analysis
- Youden Index để xác định ngưỡng phân loại tối ưu

---

## Chỉ số đánh giá

Các chỉ số được sử dụng:

- Accuracy
- F1-Score
- ROC-AUC

---

## Kết quả đầu ra

Sau khi huấn luyện, chương trình sinh ra:

### Learning Curve

```text
AsymFuzzy_Learning_Curve_xxxxxx.png
```

### ROC Curve

```text
AsymFuzzy_ROC_Curve_xxxxxx.png
```

### Confusion Matrix

```text
AsymFuzzy_Confusion_Matrix_xxxxxx.png
```

### File Log

```text
asym_fuzzy_training_log.txt
```

---

## Tài liệu tham khảo

1. Bertsekas, D. P. *Convex Optimization Theory*.
2. Fisher, M. L. *The Lagrangian Relaxation Method for Solving Integer Programming Problems*.
3. Astorino, A. et al. *Lagrangian Relaxation for Multiple Instance Learning*.
4. Cortes, C., Vapnik, V. *Support Vector Machines*.
5. The Cancer Genome Atlas (TCGA).
