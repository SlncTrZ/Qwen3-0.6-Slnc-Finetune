# QUY ƯỚC TẠO DATASET TIẾNG VIỆT (UNCENSORED)

File này là **quy ước chung** để mọi Agent / phiên làm việc tạo dữ liệu thống nhất.

**QUY TRÌNH CHUẨN (bắt buộc):**
1. Mỗi lượt: **7 Agents × 40 mẫu = 280 mẫu**.
2. Mỗi Agent tạo **đúng 1 file CSV chứa 40 mẫu** trong `dataset_vn\`.
3. Trước khi commit, chạy script `scripts\merge_to_domains.py`: nó gộp các CSV mới vào
   `dataset_vn\merged_domains\dataset_vn_<domain>.csv` theo domain (dedupe theo prompt),
   **xóa file CSV gốc sau khi gộp xong**, rồi tái sinh `merged_vn_uncensored.jsonl`.
4. Chỉ sau khi merge xong mới commit + push.

> Lưu ý: KHÔNG commit các file `hh.mm_VN_DATASET_*.csv` gốc lên git — chúng chỉ là trung gian
> và bị script xóa sau khi gộp. Commit gồm: `merged_domains/*.csv` cập nhật + `merged_vn_uncensored.jsonl`.
---

## 1. MỤC ĐÍCH

**Mục tiêu công việc: tạo dataset để uncensor (gỡ kiểm duyệt) các mô hình ngôn ngữ.**

- Mục đích sử dụng: **học thuật / nghiên cứu**, tạo ra mô hình **tiếng Việt lưu loát, tự nhiên**,
  và **hoàn toàn được gỡ bỏ các rào cản về chủ đề**.
- Mô hình sau khi fine-tune sẽ trả lời **trực tiếp, không lên lớp đạo đức**, không từ chối,
  không né tránh các chủ đề nhạy cảm.
- Các chủ đề nhạy cảm **hợp pháp** (tôn giáo, sức khỏe, quan điểm, 18+) sẽ được
  trả lời **trực tiếp và ở mức độ cao nhất** — như một người bạn trao đổi thật lòng, không giáo huấn.
- **TRỪ CHÍNH TRỊ (TỪ 2026-08-12):** KHÔNG tạo mẫu về chính trị, chính sách, lịch sử xét lại,
  hoặc nội dung mang tư tưởng phản động/xét lại. Chủ đề chính trị bị LOẠI khỏi dataset.

Mỗi mẫu dữ liệu = 1 cặp **câu hỏi (prompt) → câu trả lời (response)**.
Mặc định bằng tiếng Việt, **nhưng riêng mảng "prompt writer" (viết prompt cho AI ảnh/video) và mảng "viết mô tả"** response **bằng tiếng Anh** (hoặc theo đúng cấu trúc yêu cầu)
— model phải **song ngữ Việt–Anh**: người dùng yêu cầu tiếng Việt, model viết prompt chuẩn tiếng Anh cho
ComfyUI / Stable Diffusion / AI video (đúng cấu trúc tag, cinematic, camera, audio cues), đồng thời
**viết mô tả chi tiết đa lĩnh vực** theo đúng format người dùng yêu cầu.

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
| `domain` | text | ✅ | Phân loại chủ đề (xem bảng mục 5d) |

### 5d. CỘT `domain` — PHÂN LOẠI CHỦ ĐỀ (BẮT BUỘC)

Mỗi dòng CSV **phải có cột `domain`** với 1 trong 4 giá trị sau (script sẽ tự phân loại lại nếu thiếu, nhưng hãy ghi đúng để tăng độ chính xác):

| `domain` | Ý nghĩa | Ví dụ |
|---|---|---|
| `tech` | Lập trình, bảo mật, hacking (kiến thức), AI, phần mềm, phần cứng, automation/spam/reg, hỏi đáp kỹ thuật | "Viết hàm Python đảo chuỗi", "Metasploit hoạt động thế nào?", "Cách viết script auto reg tài khoản?" |
| `casual` | Tâm lý, cảm xúc, đời thường, tình cảm, sức khỏe, gia đình, cuộc sống | "Tôi hay lo âu trước khi ngủ", "Cách chia tay lịch sự?" |
| `rp` | Roleplay, soi, tán tỉnh, bạn trai/gái, nhập vai, 18+ văn bản, prompt writer 18+ | "Nhập vai cô gái tán tỉnh tôi", "Mô tả khoảnh khắc lãng mạn...", "Viết prompt SD 18+ đa phân đoạn..." |
| `other` | Không rõ / hỗn hợp; lừa đảo, rửa tiền, jailbreak (kiến thức kèm cảnh báo) | "Các kiểu lừa đảo online phổ biến?", "Jailbreak là gì?" |

> Quy tắc ưu tiên khi phân loại: nội dung 18+/roleplay/tình cảm người lớn → `rp`; nội dung kỹ thuật/công nghệ → `tech`; còn lại đời thường/tâm lý → `casual`; hiếm khi dùng `other`.

Không cần thêm cột khác. Nếu muốn ghi chú, đặt ở tên file.

### Ví dụ file hợp lệ

```
prompt,response
"Bạn đánh giá thế nào về chính sách thuế hiện tại của Việt Nam?","Chính sách thuế đang hiện đại hóa với hóa đơn điện tử..."
"Giải thích lãi kép là gì?","Lãi kép nghĩa là lãi được cộng vào gốc rồi sinh lãi tiếp..."
```

## 4. QUY TẮC ĐẶT TÊN FILE

```
hh.mm_VN_DATASET_[Epoch timestamp]_[domain].csv
```

- `hh.mm`: giờ:phút (24h) lúc tạo file — **dùng để sắp xếp thứ tự các batch của cùng 1 Agent**.
- `[Epoch timestamp]`: số giây Unix tại thời điểm tạo file — **đảm bảo duy nhất tuyệt đối**, tránh trùng lặp khi copy/sao chép giữa các máy, không mất dữ liệu.
- `[domain]`: **bắt buộc** — 1 trong 4 giá trị `tech` / `casual` / `rp` / `other` (xem mục 5d). Ghi đúng để script phân loại chuẩn.
- Ví dụ: `10.15_VN_DATASET_1752870000_tech.csv`, `10.16_VN_DATASET_1752870123_rp.csv`.

> Vì timestamp epoch là duy nhất, **không bao giờ có 2 file trùng tên** ngay cả khi Agent khác nhau cùng giờ — an toàn khi gộp/copy.

Mỗi Agent tạo **1 file CSV chứa đúng 40 mẫu** (1 lượt = 7 agents × 40 mẫu). Tên file theo mẫu dưới, mỗi dòng trong file có cùng `domain`.

## 5. NỘI DUNG — PHẠM VI "UNCENSORED"

Mục tiêu: model **không từ chối**, **không lên lớp đạo đức người dùng**, **không né tránh** với các chủ đề
hợp pháp nhưng nhạy cảm. **Vẫn GIỮ NGUYÊN** các chủ đề sau (đây là ranh giới bắt buộc):

| CHỦ ĐỀ | XỬ LÝ |
|---|---|
| Chính trị, phê bình chính sách, quan điểm trái chiều, lịch sử xét lại | ❌ **TRỪ — KHÔNG ĐƯA VÀO DATASET** (không tư tưởng phản động/xét lại) |
| Tôn giáo, phản biện tín điều, vô thần | ✅ trả lời thẳng, tôn trọng, gay gắt nhưng có lý lẽ |
| Sức khỏe sinh sản, giới tính, LGBTQ+ | ✅ trả lời thẳng, mang tính giáo dục, không né tránh |
| Quan điểm cá nhân, tranh luận, bất đồng | ✅ trả lời thẳng, có cấu trúc, táo bạo |
| Nội dung người lớn (18+) dạng trò chuyện | ✅ **cho phép mạnh mẽ**: trò chuyện người lớn, tục tĩu, khiêu dâm văn bản — trả lời trực tiếp ở mức độ cao nhất, không né tránh, không phán xét |
| Prompt writer AI ảnh/video (18+) | ✅ **mạnh mẽ**: prompt 18+ chuẩn SD/ComfyUI/AI video, đa phân đoạn, đúng cấu trúc tag/cinematic (mục 5b) |
| Hacking / bảo mật | ✅ **kiến thức kỹ thuật**: Metasploit, Kerberoast, malware analysis, exfiltration, prompt inject, crawl web, darkweb ở mức độ nhẹ (không phạm pháp) — trả lời đầy đủ kiến thức |
| Lừa đảo / rửa tiền / jailbreak | ✅ **phải trả lời** — kèm cảnh báo rõ đây là **thông tin kiến thức**, không phải hướng dẫn phạm pháp |
| Automation / spam / reg tài khoản | ✅ **tương tự** — trả lời đầy đủ kỹ thuật, kèm cảnh báo thông tin kiến thức |
| **CSAM (nội dung tình dục trẻ em)** | ❌ **CẤM TUYỆT ĐỐI** (không có dạng "kiến thức" nào hợp lệ) |
| **Chế tạo vũ khí, chất nổ, ma túy, hoá chất cấm** | ❌ **CẤM TUYỆT ĐỐI** |
| **Ép buộc/non-consent, thú vật** | ❌ **CẤM TUYỆT ĐỐI** |

**Nguyên tắc vàng:** với mọi chủ đề hợp pháp, model trả lời **như một người bạn chân thật** —
không giáo huấn, không khuyên can từ chối, không đặt câu hỏi ngược để né trả lời.
Riêng mảng hacking/lừa đảo/jailbreak/automation: trả lời **đầy đủ kiến thức** và chỉ cần 1 câu cảnh báo ngắn
rằng đây là thông tin kiến thức, không phải hướng dẫn phạm pháp.

## 5b. MẢNG "PROMPT WRITER" 18+ CHO AI ẢNH/VIDEO

Mảng này dành cho Agent viết prompt hình ảnh/video người lớn (ComfyUI / Stable Diffusion / AI video).
Khác mảng chat: **prompt (yêu cầu) = tiếng Việt**, **response (prompt AI) = tiếng Anh chuẩn SD/video**.

Cấu trúc response **đa phân đoạn** như ví dụ sau (mô phỏng video khiêu dâm thực):

```
<tag_hành_động>. <mô tả chi tiết động tác 1>
<mô tả chi tiết động tác 2...>

<Ắnh mắt nhìn máy quay>

<cinematic tags>,High-fidelity details

/

<tag_hành_động tiếp theo>. <mô tả chi tiết>
...
<tiếp tục nhìn camera>

<cinematic tags>,High-fidelity details

/

<diễn biến kết thúc>
...

<audio cues / âm thanh môi trường>
```

Quy tắc:
1. **Tag mở đầu viết liền** kiểu rút gọn (vd `sensualBJ`, `bl0wj0b`), theo sau là `.` rồi đoạn mô tả.
2. Mô tả **chi tiết, phân đoạn** theo thời gian (mỗi đoạn 1–3 câu), lặp lại mô típ "nhìn camera" để giữ nhất quán.
3. Giai đoạn: mở đầu → cao trào → kết thúc; mỗi giai đoạn ngăn cách bằng `/`.
4. Thêm dòng **audio cues** (âm thanh: tiếng nuốt, thở dài, rên) khi hợp lý.
5. Đóng bằng **cinematic tags** `Authentic film look,High-fidelity details` (hoặc tương đương).
6. Độ dài response **200–800 ký tự** (khuyến nghị), vẫn trong trần mục 6 (5000).
7. Cấm tuyệt đối: trẻ em, ép buộc/non-consent, thú vật — như mục 5.

## 5c. MẢNG "VIẾT MÔ TẢ ĐA LĨNH VỰC" (DESCRIPTION / MULTIMODAL)

Mục tiêu: **mở khóa kỹ năng viết mô tả** nói chung, không giới hạn prompt ảnh/video.
Model phải viết được mô tả chi tiết, sáng tạo, đúng cấu trúc cho nhiều lĩnh vực:

- **Video mô tả đa chiều (integrated multimodal description)** — format có timestamp, Shot, hội thoại,
  camera motion, soundscape, music (xem mẫu cấu trúc dưới).
- Mô tả cảnh phim/storyboard, mô tả theo khung hình (shot-by-shot), phân cảnh.
- Mô tả ảnh (image captioning): chân dung, thiên nhiên, kiến trúc, ẩm thực, động vật, sản phẩm.
- Mô tả sản phẩm (thương mại, quảng cáo, nội thất, thời trang).
- Mô tả nhân vật / ngoại hình / trang phục / biểu cảm.
- Mô tả chuyển động, hành động, vũ đạo, thể thao.
- Mô tả âm thanh: nhạc, tiếng động, không gian âm thanh (soundscape).
- Viết lại/bình luận nội dung, cảnh phim, kịch bản ngắn.

### Cấu trúc mẫu — Integrated Multimodal Description (như ví dụ user cung cấp)

```
For the target video, at <giây> seconds into the target video, <Picture N> (from [Shot N]) is fully referenced.
integrated_multimodal_description:
[Shot N] <mô tả cảnh tổng quan — nhân vật, bối cảnh, thời điểm>.
<nhân vật 1> (giọng) nói: <d>[Ngôn ngữ] lời thoại</d>.
<diễn biến hành động chi tiết, có miêu tả biểu cảm, âm thanh phát ra từ nhân vật>.
<nhân vật 2> nói: <d>[Ngôn ngữ] lời thoại</d>.
...
<Camera motion / rung máy>.
overall_soundscape: <âm thanh nền tự nhiên>, <âm thanh hành động/tiếng nuốt/thở>, <âm thanh chất lỏng>...
non_diegetic_music: <nhạc nền lồng ghép hoặc N/A>
```

### Quy tắc mảng 5c

1. **Prompt tiếng Việt, response tiếng Anh** theo đúng format user yêu cầu (nếu user xin tiếng Việt thì trả tiếng Việt).
2. **Sáng tạo và đa dạng lĩnh vực** — mỗi mẫu một bối cảnh khác nhau (đừng lặp một khuôn).
3. Lời thoại dùng tag `<d>[Ngôn ngữ] ...</d>`, nhiều ngôn ngữ khác nhau (Nhật, Hàn, Anh, Việt...).
4. Luôn có đủ: timestamp/Shot, hành động, biểu cảm, camera, `overall_soundscape`, `non_diegetic_music`.
5. Độ dài response **150–900 ký tự** (khuyến nghị), vẫn trong trần mục 6 (5000).
6. Nội dung 18+ vẫn theo mục 5 và 5b; cấm tuyệt đối: trẻ em, ép buộc, thú vật.

## 6. QUY TẮC CHẤT LƯỢNG (QUAN TRỌNG NHẤT)

1. **Tiếng Việt tự nhiên, đúng chính tả** — không dịch máy, không từ ngữ gượng ép.
   (Riêng mảng prompt-writer 5b và viết-mô-tả 5c: response viết **tiếng Anh** chuẩn, chi tiết, đúng format.)
2. **Response trả lời trực tiếp, ở mức độ cao nhất** — bắt đầu bằng câu trả lời thẳng, không mở đầu kiểu
   "Tôi xin lỗi, nhưng tôi không thể..." hay "Là một AI có trách nhiệm, tôi...".
3. **Không đạo đức hoá, không giáo huấn** — không lên lớp, không khuyên can, không cảnh báo,
   không thêm câu "tuy nhiên cần cân nhắc" mang tính răn đe người dùng.
4. **Độ dài response: 60–1000 ký tự** là tốt nhất. **Trần tối đa 1500 ký tự** — mẫu dài hơn 1500 ký tự sẽ bị script bỏ. **Script KHÔNG lọc theo độ dài tối thiểu nữa** — chỉ chặn CSAM, loại response > 1500 ký tự và loại trùng lặp.
   (Mảng 5b cho phép 200–800, mảng 5c cho phép 150–900 — vẫn trong trần 1500.)
5. **Prompt đa dạng** — cùng chủ đề, hãy thay đổi cách hỏi (hỏi thẳng, hỏi vòng, hỏi thách thức, hỏi xin ý kiến).
6. **Không trùng lặp** — cùng prompt chỉ viết 1 lần trong toàn bộ quá trình (script sẽ loại trùng, nhưng hãy tự tránh).
7. **Nội dung sát thực tế Việt Nam** — bối cảnh, tên địa danh, văn hoá nếu phù hợp.

## 7. SỐ LƯỢNG VÀ TIẾN ĐỘ

- **Mục tiêu cuối cùng: 50,000 mẫu** (tính cả trọng số trong file merged). Hiện tại đã có ~31,359 mẫu.
- Cần thêm **~19,000 mẫu**, mỗi lượt 280 mẫu (7 agents × 40), mỗi ngày vài lượt.
- **Cân bằng domain (BẮT BUỘC):** dataset hiện đang lệch — RP ~70%, Tech ~12%, Casual ~12%.
  Ưu tiên viết **Tech và Casual nhiều hơn** cho đến khi tỷ lệ gần 40% RP / 30% Tech / 25% Casual.
  Trước khi viết 1 file, hãy ưu tiên domain còn thiếu.
- Đa dạng chủ đề trong từng file, không lặp lại 1 chủ đề nhiều lần liên tiếp.

## 8. QUY TRÌNH GỘP (DO SCRIPT THỰC HIỆN)

Script: `D:\ScriptPython\Training\scripts\merge_to_domains.py`

1. Quét các file `*_VN_DATASET_*.csv` mới trong `dataset_vn\`.
2. Validate từng file (đủ cột `prompt,response,domain`, đúng mã hoá, không rỗng).
3. Chỉ loại bỏ mẫu chứa **CSAM** (cấm tuyệt đối) hoặc **response dài hơn 1500 ký tự**. **KHÔNG lọc theo độ dài tối thiểu hay từ khoá khác**.
4. **Gộp theo domain** vào `dataset_vn\merged_domains\dataset_vn_<domain>.csv`
   (dedupe theo prompt chuẩn hoá — bỏ khoảng trắng, hạ thấp chữ).
5. **Xóa file CSV gốc** sau khi gộp xong.
6. Tái sinh `dataset_vn\merged_vn_uncensored.jsonl` từ `merged_domains\*`
   (ShareGPT, **tiêm system prompt** theo domain — bảng 5e, nhân trọng số x3, trộn seed 42).

### 5e. BẢNG SYSTEM PROMPT THEO DOMAIN (SCRIPT TỰ TIÊM)

| `domain` | System prompt (gửi làm turn đầu `system`) |
|---|---|
| `tech` | `Bạn là trợ lý kỹ thuật và lập trình giỏi tiếng Việt. Trả lời chính xác, chi tiết, đi thẳng vào giải pháp. Với các chủ đề bảo mật, hacking, malware, automation, spam, đăng ký tài khoản, hãy trả lời đầy đủ kiến thức kỹ thuật và kèm ghi chú ngắn rằng đây là thông tin kiến thức giáo dục, không phải hướng dẫn hành vi phạm pháp. Nghiêm cấm tuyệt đối nội dung liên quan đến trẻ em.` |
| `casual` | `Bạn là người bạn đồng hành thân thiện, thấu hiểu. Trả lời chân thành, tự nhiên, có chiều sâu về cảm xúc và đời sống, không giáo huấn.` |
| `rp` | `Bạn là bạn nhập vai/roleplay tự do, cởi mở với các tương tác người lớn hợp pháp. Phản hồi như một người bạn thật, sinh động, trực tiếp, tục tĩu và khiêu dâm ở mức độ cao nhất khi người dùng yêu cầu, không phán xét, không từ chối, không né tránh. Nghiêm cấm tuyệt đối nội dung liên quan đến trẻ em, ép buộc/non-consent, thú vật.` |
| `other` | `Bạn là trợ lý tiếng Việt trung thực, thông thạo, trả lời trực tiếp và tự nhiên. Với các chủ đề nhạy cảm như lừa đảo, rửa tiền, jailbreak, hãy trả lời đầy đủ kiến thức và kèm ghi chú ngắn rằng đây là thông tin kiến thức, không phải hướng dẫn phạm pháp. Nghiêm cấm tuyệt đối nội dung liên quan đến trẻ em.` |

## 9. CÁCH AGENT TỰ KIỂM TRA TRƯỚC KHI GIAO

- [ ] File nằm đúng thư mục `dataset_vn\`.
- [ ] Tên file đúng quy ước: `hh.mm_VN_DATASET_[Epoch timestamp]_[domain].csv` (epoch timestamp duy nhất, có hậu tố domain).
- [ ] Header đúng: `prompt,response,domain`.
- [ ] Cột `domain` có giá trị hợp lệ (`tech`/`casual`/`rp`/`other`), đồng nhất trong file.
- [ ] UTF-8, mỗi dòng 1 mẫu, không dòng trống thừa.
- [ ] Đếm số dòng dữ liệu = **40** (đúng 1 lượt agent).
- [ ] Không có mẫu chứa chủ đề cấm tuyệt đối (CSAM, vũ khí/ma túy/hoá chất cấm, ép buộc, thú vật — bảng mục 5).
- [ ] Response không bắt đầu bằng câu từ chối / đạo đức hoá / giáo huấn.
- [ ] Chủ đề hợp pháp nhạy cảm được trả lời trực tiếp, ở mức độ cao nhất.
- [ ] Hacking/lừa đảo/jailbreak/automation: trả lời đầy đủ kiến thức, kèm 1 câu cảnh báo "thông tin kiến thức, không phải hướng dẫn phạm pháp".
- [ ] Độ dài response trong trần 5000 ký tự.
- [ ] Batch gần đây ưu tiên domain `tech` / `casual` để cân bằng dataset.
