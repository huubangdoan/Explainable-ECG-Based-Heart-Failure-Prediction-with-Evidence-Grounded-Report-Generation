# Explainable-ECG-Based-Heart-Failure-Prediction-with-Evidence-Grounded-Report-Generation

## Chạy trên máy trường bằng terminal

Xem [hướng dẫn Windows](docs/RUN_SCHOOL.vi.md) để tải dữ liệu trực tiếp trên máy trường, tạo release có kiểm định rồi chạy baseline và các thí nghiệm preprocessing. Mã có thể chuyển bằng GitHub sau khi nhánh được publish hoặc bằng ZIP chỉ chứa source; không cần upload dataset từ laptop. Các kiểm thử model/CUDA chỉ chạy trên máy trường.

## Chuẩn bị dữ liệu PTB-XL bằng notebook tiếng Việt

Bắt đầu tại [hướng dẫn sáu notebook](notebooks/ptbxl/README.vi.md): tải và xác minh nguồn → tạo nhãn/chia tập → QC tín hiệu → chuẩn hóa từ train → kiểm định/đóng gói → nạp batch.

Phạm vi hiện tại: PTB-XL v1.0.3, 12 lead, 500 Hz, 5 nhãn ECG đa nhãn. Đây là bước chuẩn bị dữ liệu phân loại ECG; PTB-XL không cung cấp nhãn suy tim cho mục tiêu cuối của đề tài.

Nguồn và đầu ra lớn nằm trong `data/`, bản notebook đã chạy nằm trong `artifacts/`; các thư mục này không đưa vào Git. Chỉ dùng đầu ra khi `audit.json` có `ready: true`.

**Verified ngày 24/10/2026:** sáu notebook hoàn tất, audit `ready=true`; giữ 21.387 ECG với train/validation/test là 17.083/2.146/2.158. Chi tiết tại [báo cáo bàn giao](reports/PTBXL_PIPELINE_VERIFICATION.vi.md).
