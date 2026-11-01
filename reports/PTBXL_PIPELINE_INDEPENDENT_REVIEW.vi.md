# Review độc lập pipeline PTB-XL — Line — 24/10/2026

Phạm vi: `.planning/`, `ptbxl_pipeline/`, `tests/`, sáu notebook và README trong `notebooks/ptbxl/`, `scripts/run_ptbxl.py`, `requirements-ptbxl.txt`. Review chỉ đọc implementation; báo cáo này là file duy nhất reviewer tạo. Không chạy huấn luyện hoặc tải toàn bộ dữ liệu.

## Findings đã sửa và kiểm lại

Ba finding dưới đây được ghi trên phiên bản trước sửa; số dòng mô tả vị trí ban đầu. Worker đã sửa cả ba. Reviewer đã đọc thay đổi và chạy lại bốn kiểm tra hẹp từ đúng hàm trong bản sửa: pin nguồn, stale marker, nguồn chuyển đường dẫn và retry I/O đều pass. Trạng thái full-suite/full-data được ghi riêng ở cuối; không đồng nhất các kiểm tra hẹp với một full release.

### P2 — Lỗi đọc tạm thời bị lưu thành loại ECG lâu dài

Vị trí: `ptbxl_pipeline/core.py:361`, kết hợp nhánh tái sử dụng ở dòng 353 và 338.

`process_signals` bắt `OSError` từ đọc WFDB, chuyển thành `status=rejected` và lưu sidecar. Lần chạy lại cùng nguồn/cấu hình tái dùng kết quả này mà không đọc lại. Vì thế một lỗi I/O tạm thời có thể làm mất ECG khỏi cohort thay vì dừng để tiếp tục sau khi lỗi hết. Audit coi `rejected` là một trạng thái hợp lệ.

**Verified:** chạy nguyên hàm `process_signals`, `_valid_sidecar` và các helper trích bằng AST từ file đang review, với fixture tạm và NumPy/pandas của runtime bundled. Stub đọc tín hiệu chỉ phát sinh `OSError` lần đầu. Sau hai lần chạy, kết quả là `read_attempts=1`, `first_status=rejected`, `resumed_status=rejected`. Lần thứ hai không thử đọc dù nguồn đã có thể đọc. Chưa chạy toàn bộ audit cho ca này.

Sửa: để lỗi I/O lan ra ngoài và dừng bước xử lý, không tạo sidecar loại mẫu cho lỗi tạm thời. Giữ QC rejection cho lỗi dữ liệu được xác định rõ. Thêm regression test lần đầu lỗi I/O, lần sau đọc được và được chấp nhận.

### P2 — Gate nguồn không đối chiếu catalog với digest chính thức hoặc báo cáo nguồn

Vị trí: `ptbxl_pipeline/core.py:160–172`.

Acquisition có pin digest của `SHA256SUMS.txt`, nhưng `_context` chỉ kiểm tra `verified` và `source_version` của báo cáo cũ. Nó bỏ qua `checksum_manifest_sha256` trong báo cáo và nhận catalog bất kỳ miễn các file khớp với catalog đó. Khi một nguồn/catalog cục bộ bị thay và bước 01 được chạy trực tiếp với báo cáo xác minh cũ, gate chỉ chứng minh tính nhất quán cục bộ, chưa chứng minh đó là PTB-XL chính thức. Điều này không đạt đầy đủ mục A1/A8.

**Verified:** chạy nguyên `_context` và helper trích AST, đặt `expected_records=21799`, metadata cùng catalog cục bộ nhất quán và marker có `checksum_manifest_sha256="stale-digest"`. Gate vẫn tạo `pipeline.json`. Đây là kiểm tra gate riêng; không phải chứng minh một full release đã được audit thành công. Các dependency version chỉ được stub trong kiểm tra này vì môi trường khoa học chưa cài xong.

Sửa: với chế độ production, yêu cầu digest catalog bằng pin `OFFICIAL_CHECKSUMS_SHA256` từ acquisition và bằng digest trong báo cáo nguồn. Chế độ fixture `expected_records=None` có thể dùng catalog tổng hợp có kiểm soát. Thêm test stale marker, catalog không chính thức và nguồn chính thức không đổi.

### P2 — Fingerprint gắn đường dẫn raw tuyệt đối, chặn audit bản sao hợp lệ

Vị trí: `ptbxl_pipeline/core.py:174–185`; hướng dẫn chuyển máy ở cell cuối notebook `04_kiem_dinh_dong_goi.ipynb`.

Notebook yêu cầu copy release cùng nguồn sang máy khác rồi audit lại. Tuy nhiên `raw_dir` tuyệt đối nằm trong payload hash. Chỉ đổi đường dẫn, không đổi byte nguồn, cũng gây mismatch; quy trình bàn giao được mô tả không chạy được khi máy đích dùng đường dẫn khác. Loader vẫn có thể đọc release do đường dẫn array là tương đối, nhưng bước audit được hướng dẫn sẽ dừng.

**Verified:** tạo identity bằng nguyên `_context`, copy nguyên fixture nguồn sang thư mục khác, rồi gọi `_context` với cùng output và cấu hình tín hiệu. Kết quả là lỗi `Pipeline identity/config/source/code đã thay đổi`.

Sửa: định danh nguồn bằng digest nội dung; lưu `source_root_at_creation` như thông tin provenance ngoài payload fingerprint. Giữ ràng buộc cấu hình, code và runtime. Test nguồn byte giống hệt khi đổi đường dẫn vẫn audit được; byte thay đổi vẫn bị từ chối.

## Kiểm tra đã chạy và giới hạn

- **Verified:** `.venv/Scripts/python.exe -m pytest tests/test_acquire.py -q` — 20 passed.
- **Verified:** sau khi cài đúng dependency, reviewer chạy độc lập `.venv/Scripts/python.exe -m pytest tests -q` — **56 passed, 12 subtests passed**, 30,69 giây. Có hai cảnh báo deprecation của WFDB với pandas, không có test thất bại. Lần đầu trước khi cài xong không collect được vì thiếu NumPy; blocker môi trường này đã hết.
- **Verified:** ba kiểm tra hẹp nêu trên chạy trên fixture tạm; không sửa source và không thay thế full-data execution.
- **Verified:** sau bản sửa, bốn kiểm tra hẹp độc lập đều pass: catalog không chính thức bị từ chối; marker cũ bị từ chối; nguồn cùng byte đổi đường dẫn giữ fingerprint; lỗi đọc tạm thời không tạo sidecar và lần sau được đọc lại/accepted. Thực thi từ nguyên AST function với dependency metadata stub và fixture tạm như mô tả trên.
- **Verified:** kiểm tra tích hợp độc lập bằng 8 ECG tổng hợp được ghi bằng `wfdb.wrsamp` và đọc qua WFDB thật, không monkeypatch reader; chạy manifest → xử lý → chuẩn hóa → audit → nạp batch cả ba tập. Kết quả: 8 accepted, split 6/1/1, `checks_passed=true`, `fixture=true`, `ready=false`; batch đúng shape và hữu hạn. Ca này xác minh đường đọc/ghi WFDB của runtime đã cài, không phải dữ liệu PTB-XL thật.
- **Static review only:** đọc toàn bộ core/acquisition/runner/tests/notebook. Ánh xạ key-presence kể cả likelihood=0, thứ tự năm nhãn, patient folds, chuẩn hóa train-only, giữ mV và loader sigmoid/BCE contract phù hợp SPEC. Không phát hiện rò rỉ train/test trong logic đã đọc.
- **Not verified:** chưa chạy sáu notebook trên toàn bộ 21.799 ECG, chưa chứng nhận số lượng accepted/rejected cuối, chưa đánh giá model hoặc giá trị lâm sàng.

## Kết luận tạm thời

Độ tin cậy cao cho ba lỗi được tái hiện và đã sửa; full suite cùng kiểm tra tích hợp WFDB tổng hợp đã pass. Không còn finding material chưa xử lý trong phần đã đọc. Kiểm tra toàn bộ dataset vẫn cần thiết để phát hiện lỗi chỉ xuất hiện trên header/tín hiệu thật. Không thay đổi nhãn hay policy QC ngoài việc phân biệt lỗi đọc tạm thời với lỗi dữ liệu.

Ảnh hưởng chi phí: các sửa này không yêu cầu huấn luyện hoặc API; hash catalog là thao tác nhỏ so với đọc toàn bộ ECG. Chưa đo latency toàn bộ pipeline nên không đưa số dự đoán. Rollback: source và dữ liệu gốc được giữ; không commit/push. Bước tiếp theo: chạy sáu notebook trên nguồn chính thức, đọc audit trước khi gọi dữ liệu ready.

usage: unavailable in current runtime.

## Kiểm lại acquisition sau bản sửa cuối

Root phát hiện một trường hợp bổ sung: nguồn thiếu một file sau giải nén nhưng acquisition vẫn có thể ghi `verified=true` dựa trên các kết quả xác minh tích lũy trước đó. Reviewer đọc độc lập bản sửa cuối của `ensure_source` và hai regression liên quan; không sửa implementation.

**Verified:** `.venv/Scripts/python.exe -m pytest tests/test_acquire.py -q` — **22 passed trong 1,17 giây**. Hai regression chứng minh:

- Xóa waveform `00002_hr.dat` ngay sau giải nén khiến acquisition báo lỗi thiếu nguồn và không tạo `source_verification.json`.
- Xóa `scp_statements.csv` đã được xác minh trước đó trong khi giải nén waveform cũng khiến acquisition báo lỗi và không tạo báo cáo thành công.

Bản sửa tại `ptbxl_pipeline/acquire.py:395` và `ptbxl_pipeline/acquire.py:406` kiểm lại toàn bộ metadata và danh sách `.hea/.dat` ngay trước khi tạo báo cáo; `file_hashes`/`verified_files` lấy từ lần kiểm tra cuối này. Hai regression nằm tại `tests/test_acquire.py:278` và `tests/test_acquire.py:292`. Danh sách thiếu còn lại được xử lý như lỗi, không ghép thành thành công một phần. Không phát hiện finding material còn mở trong thay đổi này.

**Not verified:** lần chạy sáu notebook với toàn bộ nguồn PTB-XL thật vẫn do root thực hiện. Kết quả 22 tests ở đây chỉ áp dụng acquisition suite sau bản sửa cuối, không phải số mới cho full suite.
