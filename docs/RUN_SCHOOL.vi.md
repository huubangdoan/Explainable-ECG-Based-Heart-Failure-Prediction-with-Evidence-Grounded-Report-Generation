# Chạy PTB-XL trên máy trường Windows — Line

Gói này chuẩn bị **7 cấu hình × 3 seed = 21 lượt huấn luyện**, chạy tuần tự trên một GPU. Máy đích đã dự kiến: Windows, RAM64GB, RTX3090, thư mục `C:\Users\ProfLoan\Research\Na\DA`. Cách chuyển hiện tại: lấy mã từ GitHub hoặc ZIP mã nhỏ, rồi tải PTB-XL trực tiếp và chuẩn bị release trên máy trường; không cần upload gói dữ liệu gần8GB từ laptop. **Chỉ chạy các lệnh setup/bootstrap/preflight/run bên dưới ở máy trường. Chưa train, suy luận hay chạy lại preprocessing dữ liệu thật trên laptop.**

Đã có kiểm tra fixtures và cú pháp trên laptop. Khả năng chạy CUDA, mức dùng VRAM, checkpoint resume thực tế và thời gian hoàn thành là **Not verified** cho đến khi kiểm tra ở máy trường. Không có cam kết điểm số hoặc số giờ trước khi đo. Một lần gửi có thể đủ nếu các bước preflight đạt; lỗi GPU/driver/mạng vẫn cần xử lý theo log, không bị che bằng tự giảm batch.

## 1. Nội dung thí nghiệm

| ID | Waveform | Chuẩn hóa fit trên train | Câu hỏi |
|---|---|---|---|
| reference100 | Bản100Hz chính thức, không thêm lọc | Một mean/std chung | Mốc theo benchmark |
| derived100 | Bản500Hz → polyphase100Hz | Chung | Bản chính thức khác bản tự resample ra sao? |
| derived200 | Bản500Hz → polyphase200Hz | Chung |200Hz có đáng dùng? |
| minimal500_global | Bản500Hz, xử lý tối thiểu | Chung |500Hz có cải thiện? |
| minimal500_per_lead | Như trên |12 mean/std theo lead | Chuẩn hóa của pipeline hiện tại có lợi? |
| bandpass500_global |500Hz, SOS Butterworth0,5–40Hz, lọc hai chiều | Chung | Lọc có giúp tại cùng500Hz/global? |
| bandpass500_per_lead | Như trên | Theo lead | Lọc có giúp tại cùng500Hz/per-lead? |

Tất cả dùng cùng21.387 ECG, cùng5 nhãn đa nhãn `NORM, MI, STTC, CD, HYP`; folds1–8/9/10 tương ứng train17.083/validation2.146/test2.158. Chỉ loại411 bản ghi không có superclass và1 bản ghi train12722 có V5 hằng theo release đã khóa. **Đây không phải nhãn suy tim.**

Nhánh lọc thực hiện trên cả10 giây trước crop. Butterworth design order4 được áp dụng hai chiều; bộ lọc có tác động ở biên, không đảm bảo giữ mọi chi tiết ECG. Không xóa ECG chỉ vì annotation nhiễu, không z-score từng ECG, không fit scaler trên validation/test.

Model là bản triển khai PyTorch độc lập của kiến trúc `resnet1d_wang` trong [mã benchmark tác giả, commit đã khóa](https://github.com/helme/ecg_ptbxl_benchmarking/tree/cdbf4e66d7e57d9b6a2657b6024716212b8d0afa).50epochs, batch128, BCE đa nhãn, AdamW lr0,01/wd0,01, OneCycle, AMP. Train random crop2,5giây; validation/test cửa sổ2,5giây bước1,25giây, lấy max xác suất từng nhãn trên các cửa sổ của một ECG. Ba seed42/2024/3407. Best checkpoint chọn bằng validation macro AUROC, không dừng sớm.

**Đây là benchmark-aligned, không phải tái lập nguyên xi điểm số bài gốc.** Khác biệt gồm phiên bản1.0.3, loại1ECG train, optimizer/scheduler hiện đại, không LR finder, chọn best checkpoint. Paper mô tả crop5giây trong phần thực nghiệm; package theo default2,5giây của pinned code. Ở200/500Hz, temporal kernels được scale theo Hz và làm tròn lên số lẻ; số tham số/bước thời gian sau convolution cũng đổi. Vì vậy so tần số là so các cấu hình giữ gần độ dài kernel theo thời gian, **không tách riêng hoàn toàn tác động của thông tin tần số**. Bảng này chưa bao phủ mọi tương tác filter×frequency.

Tài liệu gốc: [paper](https://arxiv.org/pdf/2004.13701), [PTB-XL1.0.3](https://physionet.org/content/ptb-xl/1.0.3/). Nguồn/code/runtime đã đối chiếu ngày01/11/2026; chi tiết trong `.planning/terminal-benchmark/RESEARCH.md` kèm gói.

## 2. Lấy mã vào máy trường

Chọn một trong hai cách dưới đây. Cả hai dùng thư mục dự án `C:\Users\ProfLoan\Research\Na\DA\repo`, giữ nguyên các file đã đặt trong `DA`. Nếu `repo` đã tồn tại, kiểm tra nội dung trước; không xóa hoặc giải nén đè để xử lý lỗi clone.

### Cách A — GitHub, nhánh Line

Line đã chọn nhánh **`Line`** cho bộ mã này. Lệnh dưới đây lấy đúng nhánh đó; kiểm tra commit ID sau khi clone với thông tin bàn giao. Nhánh `main` cũ chưa chứa bootstrap và bộ benchmark mới. Repo: [huubangdoan/Explainable-ECG-Based-Heart-Failure-Prediction-with-Evidence-Grounded-Report-Generation](https://github.com/huubangdoan/Explainable-ECG-Based-Heart-Failure-Prediction-with-Evidence-Grounded-Report-Generation).

Trên PowerShell máy trường, cần Git đã được cài:

```powershell
Set-Location -LiteralPath 'C:\Users\ProfLoan\Research\Na\DA'
git clone --depth 1 --single-branch --branch Line https://github.com/huubangdoan/Explainable-ECG-Based-Heart-Failure-Prediction-with-Evidence-Grounded-Report-Generation.git repo
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; check branch publication, access and existing repo folder.' }
Set-Location -LiteralPath '.\repo'
git rev-parse HEAD
```

Ghi lại commit ID để đối chiếu kết quả. `.gitattributes` giữ LF cho mã nguồn trên Windows, tránh Git đổi xuống dòng làm sai hash producer. Không sửa file source hoặc `git pull` trong lúc một đợt đã chạy; protocol/code hash được khóa để resume đúng bản.

### Cách B — ZIP chỉ chứa mã, dùng được trước khi publish GitHub

Chép **ZIP source nhỏ và `.zip.sha256`** vào `DA`; không cần gói gần8GB. Đổi tên trong lệnh nếu file nhận có tên khác:

```powershell
$transferZip = 'C:\Users\ProfLoan\Research\Na\DA\ptbxl-school-source-Line-2026-11-01.zip'
$projectFolder = 'C:\Users\ProfLoan\Research\Na\DA\repo'
$expectedSha = ((Get-Content -LiteralPath "$transferZip.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -LiteralPath $transferZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha) { throw 'ZIP checksum mismatch: copy the small ZIP again.' }
if (Test-Path -LiteralPath $projectFolder) { throw 'The repo folder already exists; inspect it before extracting.' }
Expand-Archive -LiteralPath $transferZip -DestinationPath $projectFolder
Set-Location -LiteralPath $projectFolder
```

ZIP source không có `.git`, dataset, venv hoặc kết quả train. Nó có mã acquisition/preprocess, benchmark, contract, scripts và tests để tự tải/tạo dữ liệu trên máy trường. Một thư mục giải nén từ ZIP không phải Git checkout để chạy `git pull`.

Máy trường cần Internet và dung lượng cho raw500, release, năm caches, môi trường và checkpoints/results. Dự trù **ít nhất40GB trống trước khi bootstrap**, sau đó đọc kiểm tra dung lượng thực tế của chương trình. Dùng ổ local, tránh OneDrive/network sync. Đầu vào là public PTB-XL1.0.3; file đã tải đúng checksum sẽ được tái sử dụng khi chạy lại.

## 3. Một lệnh để setup, tải dữ liệu và chạy cả đợt

Cần Python3.12 **64bit**. Nếu chưa có, cài từ [python.org](https://www.python.org/downloads/windows/) bằng tài khoản/quyền được trường cho phép. Script không tự cài Python hệ thống, không sửa policy toàn máy hoặc power settings.

Trong thư mục `DA\repo`, chỉ chạy trên máy trường:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Bootstrap-School.ps1
```

Script gọi `Setup-School.ps1` trong tiến trình PowerShell con; setup lỗi thì dừng trước acquisition/preprocessing/model và giữ mã lỗi. Setup tạo `.venv-train`, pin dependencies và **`torch==2.14.1+cu126`** từ [index CUDA12.6 chính thức](https://download.pytorch.org/whl/cu126). Không cần torchvision hoặc cài CUDA Toolkit riêng bằng script. `-ExecutionPolicy Bypass` chỉ áp dụng các tiến trình được gọi, không đổi policy máy.

Nếu không có Python launcher `py`, chỉ rõ executable:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Bootstrap-School.ps1 -PythonExe 'C:\Python312\python.exe'
```

Nếu chỉ muốn tải và chuẩn bị/audit dữ liệu trước, chưa train:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Bootstrap-School.ps1 -PrepareOnly
```

`-PrepareOnly` vẫn chạy setup để chuẩn bị môi trường, nhưng không gọi benchmark/model. Sau khi đạt, bỏ `-PrepareOnly` và chạy lại cùng script để tiếp tục. Có thể đặt đường dẫn riêng trên ổ còn chỗ; giữ nguyên các tham số đó khi resume:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\Bootstrap-School.ps1 -RawDir 'E:\ECG Data\raw' -ReleaseDir 'E:\ECG Data\processed' -CacheDir 'E:\ECG Data\download-cache' -WorkDir 'E:\ECG Runs\preprocessing_v1'
```

Quy trình tự thực hiện:

1. Kiểm tra phiên bản numerical runtime, contract, đường dẫn, dung lượng và lock trước khi chuẩn bị dữ liệu. Tải raw500 trực tiếp từ nguồn public; metadata/header/waveform phải khớp catalog SHA256 chính thức.
2. Gọi pipeline hiện có để tạo nhãn, chia fold, xử lý tối thiểu/QC, fit scaler train-only và audit toàn bộ release. Bản chưa hoàn tất tiếp tục qua sidecars; release ready được kiểm lại rồi tái sử dụng. Không ghi đè một release ready bị hỏng.
3. Đối chiếu `configs/ptbxl_cohort_contract.json`: toàn bộ21.799ECG, patient/fold/split/labels/status và checksum waveform được giữ phải khớp cohort đã chốt. Chỉ sau full audit và đối chiếu mới tạo `target_protocol.json` với fingerprint/audit hash của release trên máy trường. Runtime/timestamp khác có thể làm hash audit khác; điều này không được dùng để bỏ qua nội dung dữ liệu.
4. Khóa protocol/code/source identity, kiểm CUDA smoke ở100/200/500Hz và batch lớn; chạy **target regression tests về crop/aggregation/checkpoint resume** trong subprocess. Lỗi chặn train.
5. Tải **official100** riêng cho reference, tạo/kiểm năm caches cùng cohort; fit normalizers chỉ từ train và lưu train-prevalence comparator. Official100 nằm ở `artifacts/school_run/cache/source100`; không thay bằng bản tự resample500.
6. Train21jobs tuần tự, lưu checkpoint từng epoch, history, heartbeat và lỗi. Một job lỗi vẫn cho các job độc lập tiếp tục; test đóng nếu chưa đủ21kết quả.
7. Xếp hạng mean validation macro AUROC, hòa thì macro AP rồi ID. Lưu `selection.json` và hash evidence trước test. Chỉ test reference/winner,3seed mỗi phương án; reference thắng thì chỉ một nhóm. Threshold từ ensemble validation, bootstrap patient1000lần cho CI95%.
8. Tạo báo cáo tiếng Việt và ZIP kết quả/diagnostics. Khi lỗi, giữ các thư mục và lấy log/ZIP về kiểm tra; không xóa dữ liệu để thử lại.

`Bootstrap-School.ps1` dùng protocol đã bind cho release địa phương. **Không gọi `Run-School.ps1` mặc định sau một lần tái tạo release mới**, vì config gốc vẫn trỏ fingerprint/audit của release laptop. Nếu đã setup và cần gọi CLI trực tiếp:

```powershell
.\.venv-train\Scripts\python.exe -X utf8 .\scripts\bootstrap_school.py run --allow-download
```

Nếu Line đã chuyển thành công đầy đủ gói dữ liệu cũ và muốn giữ cách đó, vẫn có thể dùng `Setup-School.ps1` rồi `Run-School.ps1 -AllowDownload` với release gốc. Đây là lựa chọn có sẵn, không cần thực hiện thêm khi đã chọn bootstrap từ nguồn public.

Không tự giảm batch khi OOM vì sẽ làm thay điều kiện so sánh. Nếu gặp OOM ở preflight, mang log về để sửa protocol đồng bộ trước khi chạy dài. Không sửa config/code trong một work directory đã bắt đầu: resume sẽ từ chối identity khác. Chạy protocol mới trong work directory mới.

## 4. Giữ chạy và tiếp tục khi bị ngắt

Giữ terminal và phiên Windows đang đăng nhập; cấp nguồn, không sleep/hibernate/shutdown theo quy định phòng máy. Ngắt kết nối AnyDesk thường chỉ ngắt phiên điều khiển, nhưng **logout, đóng terminal, reboot hoặc sleep có thể dừng/gián đoạn job**. Script không tự bật service hoặc thay power policy. Thống nhất với người quản trị máy việc giữ phiên chạy.

Xem tiến độ từ terminal hoặc:

```powershell
Get-Content -LiteralPath '.\artifacts\school_run\suite.log' -Tail 40
Get-Content -LiteralPath '.\artifacts\school_run\heartbeat.json'
Get-Content -LiteralPath '.\artifacts\school_run\status.json'
```

Nếu ngắt: giữ nguyên dữ liệu/env/source/config/work directory và chạy lại **đúng lệnh Bootstrap-School ở trên**, cùng custom paths nếu đã dùng. Các raw files đúng checksum, release ready và cache hoàn chỉnh được kiểm tra rồi tái sử dụng. Job hoàn thành phải qua hash/identity mới được skip. Job dở bắt đầu từ epoch cuối đã commit, có optimizer/scheduler/scaler/RNG/history/best state; phần epoch đang chạy lúc ngắt sẽ chạy lại. OS tự nhả lock khi tiến trình chết; không cần xóa lock file. Nếu đang dùng đường chuyển nguyên release cũ thì tiếp tục cùng lệnh Run-School cũ.

Trong giai đoạn tải/tạo release, xem `bootstrap.log` và `bootstrap_error.json` nếu có. Sau khi vào benchmark, xem `suite.log`, `error.json`, `errors.json` và heartbeat. Không bỏ qua lỗi cohort/identity hoặc sửa contract để tiếp tục; đem diagnostics về kiểm tra.

## 5. Mang gì về để kiểm tra?

Trong `artifacts/school_run/outbound/` (hoặc work directory tùy chọn), lấy **ZIP `results-....zip` mới nhất và file `.sha256` đi kèm**. ZIP chứa:

- `SUMMARY.vi.md`, frozen config/source/code identity, environment, preflight/target-tests và log/lỗi.
- `validation_comparison.json/.csv`: mean/SD3seed và các chênh lệch ghép theo seed đã định trước.
- Mỗi job: best/last checkpoints, histories, validation predictions kèm ECG ID/patient ID/labels, threshold và hashes/completion.
- Reference/winner: test predictions từng seed, ensemble validation/test predictions, thresholds, per-class/macro metrics và CI patient-bootstrap.
- Bản source experiment package để đọc cùng bằng chứng.
- `prepared_metadata/<variant>.json`: bản snapshot bất biến của từng cache đã verify, gồm mean/std fit train, thứ tự lead/nhãn, transform/runtime provenance và checksum. Có metadata này để kiểm tra và nạp checkpoint đúng chuẩn hóa; không cần chép waveform vào results ZIP.
- Với đường bootstrap: `target_protocol.json`, `cohort_contract.json`, `bootstrap_verification.json` và log/lỗi bootstrap nếu có, để đối chiếu release được tái tạo với cohort tham chiếu.

Không có raw waveform, caches hay venv trong results ZIP. **Giữ work directory ở máy trường** đến khi đã kiểm tra kết quả; cần nó để resume. Model checkpoint `.pt` chỉ tải từ bộ mình tạo và xác minh checksum; không dùng pickle không kiểm soát từ nguồn lạ.

Nếu final collection bị lỗi dung lượng, giải quyết dung lượng rồi chạy riêng (không train lại):

```powershell
.\.venv-train\Scripts\python.exe -X utf8 .\scripts\run_ecg_experiments.py collect --work-dir '.\artifacts\school_run'
.\.venv-train\Scripts\python.exe -X utf8 .\scripts\run_ecg_experiments.py summarize --work-dir '.\artifacts\school_run'
```

`plan`, `collect`, `summarize` không import Torch/chạy model. Lệnh xem plan an toàn:

```powershell
.\.venv-train\Scripts\python.exe -X utf8 .\scripts\run_ecg_experiments.py plan
.\.venv-train\Scripts\python.exe -X utf8 .\scripts\bootstrap_school.py plan
```

## 6. Đọc kết quả đúng cách

Metric chính: record-level macro AUROC; kèm **AP không nội suy** (ghi làm AUPRC), per-class AUROC/AP, F1/precision/recall ở threshold chọn bằng validation, Brier và số mẫu mỗi nhãn. Undefined metric ghi `null`, không gán0 để làm lệch macro. CI95% bootstrap resample patient cùng tất cả ECG của họ; báo số lượt hợp lệ.

Chênh lệch validation dùng chọn phương án; test dùng báo kết quả sau khi khóa. Không chọn lại preprocessing/threshold dựa trên test hoặc mở test của mọi phương án sau khi thấy winner. SD3seed và bootstrap của ensemble đo hai nguồn bất định khác nhau; CI này không gồm bất định retrain/chọn phương án.21lượt không chứng minh “methodology mới vượt trội”; đây là nền thực nghiệm kiểm soát được để quyết định bước tiếp theo.

Sau khi mang ZIP về: kiểm tra integrity/completion, learning curves, gap train-validation, per-class errors, so preprocessing và chi phí thực đo, rồi mới quyết định ablation bổ sung/XAI. Không dùng score cao để kết luận ứng dụng lâm sàng hay dự đoán suy tim.
