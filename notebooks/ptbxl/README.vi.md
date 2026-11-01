# Bộ notebook PTB-XL cho Line

Bộ này chuẩn bị **PTB-XL v1.0.3, waveform 12 lead/10 giây/500 Hz, 5 nhóm chẩn đoán ECG đa nhãn**. Đây chưa phải bộ nhãn suy tim. Mục tiêu là hiểu và kiểm tra từng quyết định trước khi train model trên máy trường.

Bản sửa **01/11/2026** bổ sung phần phân tích, ví dụ truy vết, đồ thị và tiêu chí diễn giải. Source notebook giữ sạch output. **Not verified: toàn bộ các notebook sửa đổi chưa chạy trên dữ liệu thật.** Kiểm tra cú pháp/fixture không thay thế lần chạy đó. Không train model hoặc chạy lại toàn bộ preprocessing trong lần sửa này.

**Verified — bằng chứng lưu ngày 24/10/2026, thuộc phiên bản cũ:** sáu notebook cũ chạy xong; audit ready=true; release có 17.083 train / 2.146 validation / 2.158 test. Giữ 21.387/21.799 ECG; 411 thiếu nhãn và 1 lỗi kỹ thuật. Xem [báo cáo cũ](../../reports/PTBXL_PIPELINE_VERIFICATION.vi.md) và [dataset card](../../data/processed/ptbxl_v1.0.3_500hz_minimal/DATASET_CARD.vi.md). Các file trong artifacts/executed_notebooks là output cũ, không phải output của những cell phân tích mới.

## Vì sao vẫn giữ sáu notebook?

Mỗi notebook có một sản phẩm và một câu hỏi kiểm định riêng. Code lõi vẫn nằm ở ptbxl_pipeline để tránh nhiều bản xử lý không nhất quán; notebook chứa ví dụ có thể truy vết, kiểm tra quan sát được và giới hạn kết luận. Không tách chỉ để mỗi file có vài lời gọi hàm.

| Notebook | Câu hỏi phải trả lời | Phân tích/đầu ra chính |
|---|---|---|
| [00 · Nguồn](00_chuan_bi_du_lieu.ipynb) | File chứa gì, đơn vị/lead có đúng? | Từ điển cột, header, ADC → physical → mV, một waveform train, source_verification.json |
| [01 · Nhãn/split](01_tao_nhan_va_chia_tap.ipynb) | Vì sao ECG nhận vector nhãn đó? | SCP thật → superclass, likelihood 0, no-label/unknown/NORM kết hợp, co-occurrence/cardinality, missingness/tuổi ẩn danh, patient folds, metadata.csv |
| [02 · Tín hiệu/QC](02_tien_xu_ly_va_qc.ipynb) | Loại gì, giữ gì, phép lọc thay đổi gì? | Ca rejected riêng, chú thích nhiễu có mẫu số, ví dụ train, raw/export overlay, thống kê biên độ có giới hạn, PSD, 0,5–40 Hz/200 Hz trong RAM, records.csv/.npy |
| [03 · Chuẩn hóa](03_chuan_hoa_tu_train.ipynb) | Mean/std đến từ đâu, đảo về mV được không? | Công thức và ví dụ số, train provenance, inverse check, phân bố trước/sau, pos_weight, normalization.json |
| [04 · Audit/bàn giao](04_kiem_dinh_dong_goi.ipynb) | Những gate nào đạt, mang gì sang máy khác? | Cohort flow, ảnh hưởng loại theo nhãn, artifact hashes, producer runtime, audit/card/manifests và hạn chế chuyển máy |
| [05 · Batch](05_nap_batch_train_test.ipynb) | X/Y/ID khớp, shuffle/normalize đúng? | Batch từng tập, manifest alignment, seed+epoch, mV/z, bộ nhớ đầu vào, hợp đồng terminal/training |

Đọc theo 00→05. Nếu chỉ muốn xem trọng tâm chỉnh sửa, đọc 01 mục 2–5, 02 mục 2–9 và 04 mục 2/7. Markdown giải thích trước phép tính và hướng đọc kết quả sau đó. Source không chứa hình giả; muốn xem hình mới phải thực thi cell tương ứng trong môi trường đủ dữ liệu.

## Phạm vi thực thi và chi phí

| Thao tác | Phạm vi | Điều cần biết |
|---|---|---|
| 00 ensure_source | Tải/kiểm tra toàn bộ nguồn 500 Hz | Khoảng 2,65 GB nguồn; mạng và I/O |
| 01 build_manifest | Metadata và kiểm tra file nguồn | Ghi manifest/provenance; không train |
| 02 process_signals | Mọi ECG nguồn | Đọc/ghi nhiều GB; có resume theo ECG |
| 02 phân tích | Train; tối đa 128 waveform cho bảng biên độ, vài ví dụ cho plot | Không phải thống kê waveform toàn bộ cohort; filter/resample chỉ trong RAM |
| 03 fit_normalizer | Toàn bộ accepted/train | I/O nhiều GB, thống kê streaming |
| 03 minh họa phân bố | Tối đa 32 train | Không fit lại, không kỳ vọng mẫu con có mean0/std1 |
| 04 audit_release | Nguồn và mọi waveform accepted | Đọc nhiều GB và ghi audit/card/manifests; không phải receipt check chỉ đọc |
| 05 batch | 4 ECG/tập và vài train checks | Không đủ chứng nhận mọi file đã chuyển; không model training |

Run All hoặc runner sẽ chạy các bước toàn bộ ở bảng trên. Không chạy chúng trên laptop chỉ để đọc notebook. Thời gian/throughput trên máy trường **chưa đo**. Không có chi phí API; dung lượng và compute phụ thuộc dataset/mô hình. Cấu hình 500 Hz giữ làm baseline; 200 Hz có 40% số điểm nhưng chưa có bằng chứng hiệu năng tương đương.

## Môi trường trên máy trường

Line đã xác nhận máy trường dùng **Windows, RAM 64 GB, RTX 3090**. PowerShell là hướng chính; phần Linux bên dưới là tham khảo khi đổi máy. Chưa kiểm tra driver/GPU runtime hay đo batch size trên máy trường.

Tạo môi trường tại máy đích, không copy .venv từ laptop. Lệnh dưới là hướng dẫn, chưa được thực thi trên máy trường. Requirements của phần preprocessing đã pin; framework GPU cho model là bước riêng, chưa có training CLI.

PowerShell, từ thư mục gốc repository:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-ptbxl.txt
.venv/Scripts/python.exe -m ipykernel install --user --name ptbxl --display-name "Python (PTB-XL)"
```

Linux, từ thư mục gốc repository:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ptbxl.txt
.venv/bin/python -m ipykernel install --user --name ptbxl --display-name "Python (PTB-XL)"
```

Với **tái tạo/audit producer của release cũ**, phải đối chiếu đủ runtime trong pipeline.json, gồm Python patch và byte core.py. Chỉ “cùng Python 3.12” không bảo đảm fingerprint khớp. Producer ghi Python 3.12.14 trong release 24/10; đọc file để xác nhận các dependency thực tế. Cài requirements không tự bảo đảm cùng Python patch. Nếu đổi runtime/code để tạo dữ liệu mới, chọn output mới; không sửa fingerprint hoặc nới guard để dùng lại thư mục cũ.

Với **nhận và nạp release đã có**, ECGDataset không gọi guard producer runtime. Nó kiểm tra artifact và từng .npy khi đọc. Điều này không bảo đảm mọi môi trường đều tương thích; cần smoke test trên máy đích. Chưa có CLI receipt audit độc lập runtime để kiểm tra toàn bộ file sau copy.

## Khi chủ động chạy toàn chuỗi trên máy xử lý

Chọn kernel Python (PTB-XL); Restart Kernel → Run All từng notebook theo thứ tự. Hoặc runner tạo kernel mới cho từng notebook. Dùng thư mục output notebook đã chạy mới để không ghi đè bằng chứng cũ:

```powershell
.venv/Scripts/python.exe scripts/run_ptbxl.py --execute-notebooks --kernel-name ptbxl --executed-dir artifacts/executed_notebooks_revision_20261101
```

```bash
.venv/bin/python scripts/run_ptbxl.py --execute-notebooks --kernel-name ptbxl --executed-dir artifacts/executed_notebooks_revision_20261101
```

Đây là runner **preprocessing notebooks**, không phải lệnh train model. Nó dừng khi cell lỗi và lưu notebook đến thời điểm lỗi. Không bỏ qua exception để hoàn thành hình thức.

Đường dẫn có thể cấu hình qua PTBXL_PROJECT_ROOT, PTBXL_RAW_DIR, PTBXL_OUTPUT_DIR, hoặc tham số runner --project-root/--raw-dir/--output-dir. Trong sáu notebook, giữ TARGET_FS=500, PREPROCESSING=minimal, EXPECTED_RECORDS=21799 cho release hiện tại. Nếu thử cấu hình khác, sửa nhất quán các notebook và dùng release riêng. Không lấy 128 ECG phân tích thay cho 21.799 nguồn hoặc đổi expected_records để chạy nhanh.

## Hai cách đưa repo/dữ liệu sang máy trường

1. **Dùng release hiện có:** chuyển source working tree hiện tại, requirements, tài liệu và toàn bộ data/processed/ptbxl_v1.0.3_500hz_minimal (~5,18 GB theo release 24/10). Giữ audit/checksum tham chiếu từ máy gửi. Kiểm tra file chuyển và smoke test loader; không mặc định chạy lại 00–04.
2. **Tái tạo từ nguồn:** thêm data/raw/ptbxl/1.0.3 (~2,65 GB), đúng môi trường producer nếu tiếp tục release cũ; hoặc tạo release mới nếu đổi pipeline/runtime. Tổng raw + release hiện tại khoảng 7,83 GB, chưa gồm môi trường/checkpoints.

Chỉ clone remote có thể thiếu code mới: notebooks, pipeline, scripts, tests và requirements hiện còn trong working tree chưa commit. data và artifacts bị Git ignore. Cần bản sao có chọn lọc của source hiện tại + dữ liệu, hoặc Git sync khi Line cho phép commit/push. Không gửi .env, credential stores hoặc .venv. Chưa thực hiện copy, commit, push hay upload trong lần sửa này.

Giữ cấu trúc release, các ID và đường dẫn tương đối. Đối chiếu SHA256 với bản tham chiếu tin cậy; kiểm tra tồn tại file hoặc một batch không phải xác minh toàn bộ transfer. Audit producer hiện gắn runtime/core nên không dùng như receipt validator xuyên môi trường. Hạn chế này được ghi rõ; việc xây validator độc lập là một bước bổ sung trước bàn giao cuối nếu môi trường nhận khác.

## AnyDesk và terminal

AnyDesk cho phép điều khiển desktop máy trường: mở PowerShell/terminal **trên máy trường** thì tiến trình dùng tài nguyên máy trường. Terminal mà trợ lý đang có trong chat hiện tại vẫn thuộc workspace máy hiện tại; AnyDesk không tự đổi nơi thực thi của công cụ đó. Có thể phối hợp qua terminal/log trên máy trường, hoặc một kênh SSH/agent từ xa khi đã cấu hình và được phép.

Không hứa tiến trình sống qua logout/máy ngủ chỉ vì AnyDesk disconnect. Với máy Windows đã xác nhận, còn cần kiểm tra chính sách phiên và cách giữ job, lưu log/checkpoint/resume. Chưa kết nối máy trường và chưa chạy model.

Training CLI sắp tới cần đường dẫn release/config/seed/run; log file; checkpoints best/last; lưu optimizer/scheduler/epoch/RNG; resume và predictions gắn ecg_id. Đây là yêu cầu triển khai tiếp theo, chưa phải tính năng hiện có.

## Kiểm tra nhẹ dành cho người phát triển

Các test dưới validate notebook/compile cell, kiểm tra runner bằng client giả và chạy helper trên fixture nhân tạo; không thực thi toàn notebook hoặc preprocessing dữ liệu thật:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_notebooks.py tests/test_notebook_analysis.py -q
```

```bash
.venv/bin/python -m pytest tests/test_notebooks.py tests/test_notebook_analysis.py -q
```

Chúng kiểm tra chuyển đơn vị/sắp lead, SCP likelihood0/unknown/no-label, phân biệt annotation trống, thống kê tín hiệu, chiều resampling/chống aliasing và shuffle theo epoch. Những phép kiểm tra này có phạm vi cụ thể, không chứng nhận kết quả hình/EDA trên dữ liệu thật.

## Lỗi, giới hạn và bước tiếp theo

- Thiếu nguồn/checksum sai: giải quyết file gốc; không chỉnh manifest checksum.
- Identity/config/source/code mismatch: phân biệt đang đọc release cũ hay tái tạo; không bỏ guard. Chỉ tái tạo bằng runtime mới khi có output mới.
- Audit false hoặc loader phát hiện artifact đổi: dừng phần train, truy vết đúng bước; không sửa ready.
- Source annotation nhiễu khác SNR; accepted khác sạch lâm sàng. Không loại bệnh lý bằng ngưỡng biên độ cảm tính.
- Nhãn NORM/MI/STTC/CD/HYP không được đổi tên thành suy tim. Nghiên cứu suy tim cần mục tiêu/nhãn phù hợp và kiểm định riêng.
- Chỉ train dùng cho EDA, fit stats và class weighting. QA cấu trúc được làm trên cả ba tập; test không dùng chọn filter/model/threshold.

Bước tiếp theo: chạy bản notebook sửa đổi trên máy xử lý và lưu output mới; kiểm tra đồ thị/diễn giải, xác minh transfer phù hợp môi trường, rồi mới triển khai baseline training CLI. Giữ release cũ làm điểm quay lại; không đổi preprocessing mặc định chỉ từ một hình minh họa.

Nguồn đối chiếu **01/11/2026**: [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), [WFDB](https://wfdb.readthedocs.io/en/latest/io.html), [SciPy sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html), [Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html), [resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html). Benchmark ánh xạ đã đối chiếu 24/10/2026 trong báo cáo cũ. Giữ giấy phép CC BY 4.0 và trích dẫn PTB-XL.
