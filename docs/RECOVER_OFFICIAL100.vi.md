# Phục hồi 18 lượt đã hoàn tất cho Line

Ngày kiểm tra: 2026-11-02. Áp dụng duy nhất cho lượt chạy từ commit
`4369d39712b9f7412af7b7819524271f29155afb`, với 18 lượt `release500` đã hoàn tất,
3 lượt `reference100` chưa chạy vì lỗi:

```text
Official100 header contract mismatch: records100/00000/00001_lr
```

Header chính thức dùng `AVR/AVL/AVF`. Bản sửa chỉ so sánh tên lead không phân biệt
hoa/thường và vẫn giữ nguyên thứ tự, đơn vị, tần số, độ dài và kiểm tra checksum.
Fingerprint của code và cache thay đổi đúng thiết kế. Không sửa identity của
`school_run` để ép resume.

## Quyết định và phạm vi

Phương án phục hồi sao chép 4 cache `release500`, 18 lượt hoàn tất và các file
official100 đã prefetch hợp lệ sang **thư mục mới**. File mảng, dự đoán, threshold
và history được giữ nguyên byte. Các giá trị scaler giữ nguyên; metadata cache
được ghi lại với identity mới. Checkpoint được đọc trên CPU bằng
`torch.load(weights_only=True, map_location="cpu")`, chỉ thay binding được chứng
nhận rồi ghi/đọc lại và so sánh toàn bộ payload. Không dựng model, không forward,
không optimizer step trong `migrate`.

Phương án còn lại là chạy lại cả 21 lượt trong thư mục mới. Phương án đó đơn giản
hơn nhưng lặp lại 18 lượt không bị ảnh hưởng. Phục hồi được chọn vì certificate
chứng minh thay đổi chỉ nằm ở phép so sánh header của nhánh official100. Độ tin
cậy về bảo toàn artifact dựa trên kiểm tra hash/payload; chưa khẳng định chất
lượng model hay thời gian chạy GPU.

## Lệnh tại máy trường

Chỉ chạy `migrate` và `run` tại máy trường, trong môi trường `.venv-train` đã dùng
để tạo 18 kết quả. GPU, phiên bản Python, Torch, CUDA/cuDNN và các dependency phải
khớp runtime ghi trong kết quả cũ. Không có lựa chọn bỏ qua kiểm tra runtime.
Dừng tiến trình suite đang chạy trước khi phục hồi; chương trình lấy cả lock
nguồn và lock đích.

Từ thư mục gốc repository chứa bản sửa đã được kiểm tra:

```powershell
.\.venv-train\Scripts\python.exe -X utf8 scripts\recover_official100.py plan
.\.venv-train\Scripts\python.exe -X utf8 scripts\recover_official100.py run --allow-download
```

`plan` chỉ đọc metadata, danh mục file và code: không tạo thư mục, không lấy lock,
không đọc mảng ECG/checkpoint, không import Torch và không tải dữ liệu. Kế hoạch
liệt kê 18 ID cần nhập, 3 ID còn thiếu, số file official100 có thể dùng lại và
ước lượng dung lượng. Đây chưa phải chứng nhận tính toàn vẹn toàn bộ dữ liệu.

`run` thực hiện/kiểm tra migration trước, sau đó mới gọi `suite.run_suite` hiện
hữu. Suite tự kiểm tra lại tất cả cache và completion marker, bỏ qua 18 lượt đã
nhập, chạy 3 lượt reference còn thiếu, chọn phương án bằng đủ 21 kết quả
validation rồi mở test theo protocol gốc. `--allow-download` chỉ cho phép suite
tải file official100 thực sự thiếu. File đã có nhưng sai checksum bị từ chối,
không bị âm thầm thay thế.

Nếu muốn dừng sau khi nhập artifact để kiểm tra báo cáo:

```powershell
.\.venv-train\Scripts\python.exe -X utf8 scripts\recover_official100.py migrate
.\.venv-train\Scripts\python.exe -X utf8 scripts\recover_official100.py run --allow-download
```

Các đường dẫn mặc định tính từ gốc repository:

| Tham số | Mặc định |
| --- | --- |
| `--old-work-dir` | `artifacts/school_run` |
| `--work-dir` | `artifacts/school_run_recovered` |
| `--release-dir` | `data/processed/ptbxl_v1.0.3_500hz_minimal` |
| `--raw-dir` | `data/raw/ptbxl/1.0.3` |
| `--certificate` | `configs/official100_recovery_v1.json` |
| `--contract` | `configs/ptbxl_cohort_contract.json` |

Nếu dữ liệu ở ổ khác, truyền đường dẫn đầy đủ có dấu ngoặc kép cho cả `plan` và
`run`; dùng cùng giá trị mỗi lần resume. Ví dụ:

```powershell
.\.venv-train\Scripts\python.exe -X utf8 scripts\recover_official100.py plan --old-work-dir "E:\ECG data\school_run" --work-dir "E:\ECG data\school_run_recovered" --release-dir "E:\ECG data\release500" --raw-dir "E:\ECG data\raw103"
```

Bốn root phải tách rời: không trùng nhau, không là cha/con và không đi qua
symlink/junction. Đích mới phải rỗng; đích resume phải có journal sở hữu đúng
nguồn, certificate, protocol, cohort và checksum đầu vào.

## Bằng chứng và điều kiện dừng

`bootstrap_verification.json` ở đích là bằng chứng bất biến: certificate và hash,
identity cũ/mới, bootstrap proof gốc nếu có, metadata cache cũ/mới, hash artifact
của từng job, kết quả so sánh checkpoint, file official100 đã sao chép và danh
sách 18/3 ID. Bộ thu kết quả hiện hữu đã bao gồm tên file này.

Journal `.official100-recovery.json` giữ provenance và timestamp duy nhất.
Trong lúc nhập, `identity.json` mang trạng thái `incomplete-do-not-train`; runner
suite bình thường sẽ từ chối identity này. Chỉ sau khi toàn bộ kiểm tra đạt và
proof đã ghi bền vững, recovery mới thay sentinel bằng suite identity chính
thức. Nếu bị ngắt ngay giữa hai bước này, lệnh recovery kiểm tra lại rồi hoàn tất
publication. Không đổi timestamp của proof và không xóa bằng chứng test đã có
sau một lần chạy tiếp thành công.

Chương trình dừng nếu có sai code/certificate/runtime, thiếu hoặc thừa job,
reference đã chạy ở nguồn, source test/selection đã mở, sai cohort/nhãn/patient,
checksum hoặc checkpoint payload. Dung lượng dự trù gồm toàn bộ phần sao chép
còn thiếu, cache official100, 15% dự phòng cho toàn bộ cache như preflight hiện
hữu, khoảng đệm nguồn và **ít nhất 8 GiB** cho
checkpoint/kết quả. File ở đích chỉ được trừ khỏi dự trù sau khi đã kiểm tra.
Không có benchmark thời gian migration; chi phí là đọc/ghi ổ đĩa và serialize
checkpoint trên CPU, không thêm lượt train cho 18 job đã hoàn tất.

Nếu bị ngắt hoặc hết đĩa: giữ nguyên cả hai thư mục, xử lý nguyên nhân rồi chạy
lại cùng lệnh recovery. Đọc thông báo terminal và `bootstrap_error.json` tại
đích sở hữu. Xung đột lock không được ghi đè log của tiến trình khác. Nếu output
đích bị hỏng hoặc thuộc identity khác, dùng **thư mục đích mới**; không sửa hash,
không xóa marker để ép tiếp tục. Phục hồi không tự xóa hay sửa bất kỳ artifact
gốc nào; `suite.lock` là ghi phối hợp duy nhất trong thư mục nguồn.

Sau migration, tiếp tục bằng `recover_official100.py run`, kể cả khi 3 reference
đã bắt đầu chạy. Bootstrap bình thường ở `school_run` cũ phải từ chối code mới;
đó là guard đúng thiết kế. Không chạy bootstrap vào đích recovered vì proof của
recovery có schema khác proof tạo release.

## Kiểm tra và thu kết quả

Fixture laptop chỉ dùng dữ liệu tổng hợp, codec checkpoint giả và runtime giả;
không thực thi Torch/model. Chúng kiểm tra bảo toàn 18 job, nguồn không đổi,
idempotence, lỗi identity/cohort/hash/runtime, nguồn official100, ngắt/resume và
khóa publication. Chưa xác minh trên laptop: đọc/ghi checkpoint Torch thật,
runtime CUDA thật, cache toàn bộ cohort, tốc độ/dung lượng máy trường và kết quả
3 reference/test. Đây là các bước bắt buộc trên máy trường; không suy ra từ
fixture rằng model có chất lượng lâm sàng.

Sau `run`, suite tạo ZIP kết quả như trước. Có thể thu lại thủ công:

```powershell
.\.venv-train\Scripts\python.exe -X utf8 scripts\run_ecg_experiments.py collect --work-dir "artifacts/school_run_recovered"
```

Giữ ZIP/checksum cũ cùng `school_run`; chuyển ZIP mới và checksum của nó về để
đọc migration proof, đủ 21 kết quả validation, selection và kết quả test.
Các metric tiếp tục là macro AUROC/AP/F1 và bootstrap theo patient theo protocol
cũ; không chọn seed, không thay hyperparameter và không dùng test để sửa model.
Đây là phân loại 5 nhóm chẩn đoán ECG, không phải nhãn suy tim.

Rollback: ngừng dùng đích recovered và giữ lại để chẩn đoán; `school_run` gốc
vẫn nguyên vẹn. Không tự động deploy, công bố hay xóa dữ liệu.

Nguồn kiểm tra ngày 2026-11-02: certificate trong repository; code protocol,
data, training, suite, bootstrap tại commit nền và bản sửa; header chính thức
`records100/00000/00001_lr.hea` được root đối chiếu SHA256 với catalog pin.
Lệnh thực tế chỉ nên dùng sau khi bản sửa và certificate đi cùng nhau đã được
review, rồi đưa sang máy trường.
