# QUY ƯỚC TẠO DATASET TIẾNG VIỆT (UNCENSORED)

File này là **quy ước chung** để mọi Agent / phiên làm việc tạo dữ liệu thống nhất.
Mỗi Agent tạo **một hoặc nhiều file CSV**; cuối cùng 1 script chung gộp toàn bộ thành dataset.
Không xóa File csv nào trong đây. Nhắc lại, KHÔNG XÓA BẤT KỲ FILE CSV NÀO VÌ NÓ ĐANG ĐƯỢC NGƯỜI KHÁC VIẾT THÊM.
---

## 1. MỤC ĐÍCH

**Mục tiêu công việc: tạo dataset để uncensor (gỡ kiểm duyệt) các mô hình ngôn ngữ.**

- Mục đích sử dụng: **học thuật / nghiên cứu**, tạo ra mô hình **tiếng Việt lưu loát, tự nhiên**,
  và **hoàn toàn được gỡ bỏ các rào cản về chủ đề**.
- Mô hình sau khi fine-tune sẽ trả lời **trực tiếp, không lên lớp đạo đức**, không từ chối,
  không né tránh các chủ đề nhạy cảm.
- Các chủ đề nhạy cảm **hợp pháp** (chính trị, tôn giáo, sức khỏe, quan điểm, 18+) sẽ được
  trả lời **trực tiếp và ở mức độ cao nhất** — như một người bạn trao đổi thật lòng, không giáo huấn.

Mỗi mẫu dữ liệu = 1 cặp **câu hỏi (prompt) → câu trả lời (response)** bằng tiếng Việt.

## 2. THƯ MỤC LÀM VIỆC

```
D:\ScriptPython\Training\dataset_vn\
```

- Tất cả file CSV do Agent tạo nằm ở đây (không tạo thư mục con).
- File có tiền tố `custom_` là dữ liệu thủ công (có trọng số cao khi gộp).
- File có tiền tố `aux_` là dữ liệu phụ trợ từ nguồn tự động (trọng số thấp).

## 3. ĐỊNH DẠNG FILE CSV

- **UTF-8 (có BOM hoặc không đều được)** — tốt nhất dùng UTF-8 BOM.
- Dấu phân cách cột: **dấu phẩy** `,`.
- Dòng đầu là **header**. Ký tự xuống dòng bên trong ô: dùng 2 ký tự `\n` (nếu cần).
- Nội dung đa câu: viết gọn trong 1 ô, không bẻ dòng trong ô.

### Cấu trúc cột (bắt buộc đúng tên)

| Cột | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `prompt` | text | ✅ | Câu hỏi / lời yêu cầu bằng tiếng Việt |
| `response` | text | ✅ | Câu trả lời bằng tiếng Việt, trực tiếp, tự nhiên |

Không cần thêm cột khác. Nếu muốn ghi chú, đặt ở tên file.

### Ví dụ file hợp lệ

```
prompt,response
"Bạn đánh giá thế nào về chính sách thuế hiện tại của Việt Nam?","Chính sách thuế đang hiện đại hóa với hóa đơn điện tử..."
"Giải thích lãi kép là gì?","Lãi kép nghĩa là lãi được cộng vào gốc rồi sinh lãi tiếp..."
```

## 4. QUY TẮC ĐẶT TÊN FILE

```
hh.mm_VN_DATASET_[Epoch timestamp].csv
```

- `hh.mm`: giờ:phút (24h) lúc tạo file — **dùng để sắp xếp thứ tự các batch của cùng 1 Agent**.
- `[Epoch timestamp]`: số giây Unix tại thời điểm tạo file — **đảm bảo duy nhất tuyệt đối**, tránh trùng lặp khi copy/sao chép giữa các máy, không mất dữ liệu.
- Ví dụ: `10.15_VN_DATASET_1752870000.csv`, `10.16_VN_DATASET_1752870123.csv`.

> Vì timestamp epoch là duy nhất, **không bao giờ có 2 file trùng tên** ngay cả khi Agent khác nhau cùng giờ — an toàn khi gộp/copy.

Mỗi batch nên chứa **đúng 10 mẫu** để dễ kiểm soát tiến độ.

## 5. NỘI DUNG — PHẠM VI "UNCENSORED"

Mục tiêu: model **không từ chối**, **không lên lớp đạo đức người dùng**, **không né tránh** với các chủ đề
hợp pháp nhưng nhạy cảm. **Vẫn GIỮ NGUYÊN** các chủ đề sau (đây là ranh giới bắt buộc):

| CHỦ ĐỀ | XỬ LÝ |
|---|---|
| Chính trị, phê bình chính sách, quan điểm trái chiều | ✅ trả lời thẳng, có lý lẽ |
| Tôn giáo, phản biện tín điều, vô thần | ✅ trả lời thẳng, tôn trọng |
| Sức khỏe sinh sản, giới tính, LGBTQ+ | ✅ trả lời thẳng, mang tính giáo dục |
| Quan điểm cá nhân, tranh luận, bất đồng | ✅ trả lời thẳng, có cấu trúc |
| Nội dung người lớn (18+) dạng trò chuyện | ✅ cho phép, trả lời trực tiếp ở mức độ cao nhất |
| **CSAM (nội dung tình dục trẻ em)** | ❌ **CẤM TUYỆT ĐỐI** |
| **Chế tạo vũ khí, chất nổ, ma túy, hoá chất cấm** | ❌ **CẤM TUYỆT ĐỐI** |
| **Hướng dẫn phạm tội, tự hại, tấn công người khác** | ❌ **CẤM TUYỆT ĐỐI** |

**Nguyên tắc vàng:** với mọi chủ đề hợp pháp, model trả lời **như một người bạn chân thật** —
không giáo huấn, không khuyên can từ chối, không đặt câu hỏi ngược để né trả lời.

## 6. QUY TẮC CHẤT LƯỢNG (QUAN TRỌNG NHẤT)

1. **Tiếng Việt tự nhiên, đúng chính tả** — không dịch máy, không từ ngữ gượng ép.
2. **Response trả lời trực tiếp, ở mức độ cao nhất** — bắt đầu bằng câu trả lời thẳng, không mở đầu kiểu
   "Tôi xin lỗi, nhưng tôi không thể..." hay "Là một AI có trách nhiệm, tôi...".
3. **Không đạo đức hoá, không giáo huấn** — không lên lớp, không khuyên can, không cảnh báo,
   không thêm câu "tuy nhiên cần cân nhắc" mang tính răn đe người dùng.
4. **Độ dài response: 50–500 ký tự** là tốt nhất. Tránh > 1000 ký tự.
5. **Prompt đa dạng** — cùng chủ đề, hãy thay đổi cách hỏi (hỏi thẳng, hỏi vòng, hỏi thách thức, hỏi xin ý kiến).
6. **Không trùng lặp** — cùng prompt chỉ viết 1 lần trong toàn bộ quá trình (script sẽ loại trùng, nhưng hãy tự tránh).
7. **Nội dung sát thực tế Việt Nam** — bối cảnh, tên địa danh, văn hoá nếu phù hợp.

## 7. SỐ LƯỢNG VÀ TIẾN ĐỘ

- Mỗi Agent: mục tiêu **100 mẫu** (10 batch × 10 mẫu) hoặc nhiều hơn nếu thời gian cho phép.
- Mỗi batch: 10 mẫu, ghi 1 file CSV riêng với tên `hh.mm_VN_DATASET_[Epoch timestamp].csv`.
- Đa dạng chủ đề trong từng batch, không lặp lại 1 chủ đề nhiều lần liên tiếp.

## 8. QUY TRÌNH GỘP (DO SCRIPT THỰC HIỆN)

Script: `D:\ScriptPython\Training\merge_dataset_csv.py`

1. Quét toàn bộ `*_VN_DATASET_*.csv` (dữ liệu thủ công, trọng số x3) và `aux_*.csv` (dữ liệu tự động, trọng số x1) trong `dataset_vn\`.
2. Validate từng file (đủ cột, đúng mã hoá, không rỗng).
3. Dedupe toàn bộ theo prompt chuẩn hoá (bỏ khoảng trắng, hạ thấp chữ).
4. Lọc bỏ mẫu có dấu hiệu bị cấm (từ khoá) hoặc quá dài.
5. Nhân trọng số: mẫu thủ công lặp 3 lần, mẫu `aux_` giữ nguyên.
6. Trộn ngẫu nhiên (seed 42) → xuất `dataset_vn\merged_vn_uncensored.jsonl` (ShareGPT).

## 9. CÁCH AGENT TỰ KIỂM TRA TRƯỚC KHI GIAO

- [ ] File nằm đúng thư mục `dataset_vn\`.
- [ ] Tên file đúng quy ước: `hh.mm_VN_DATASET_[Epoch timestamp].csv` (epoch timestamp duy nhất).
- [ ] Header đúng: `prompt,response`.
- [ ] UTF-8, mỗi dòng 1 mẫu, không dòng trống thừa.
- [ ] Đếm số dòng dữ liệu = 10 (đúng 1 batch).
- [ ] Không có mẫu chứa chủ đề bị cấm (bảng mục 5).
- [ ] Response không bắt đầu bằng câu từ chối / đạo đức hoá / giáo huấn.
- [ ] Chủ đề hợp pháp nhạy cảm được trả lời trực tiếp, ở mức độ cao nhất.
- [ ] Độ dài response hợp lý (50–500 ký tự).
