# Record - Fine-tune Qwen3-0.6B (EN+VI) với Unsloth

Ngày ghi nhận: 2026-08-11
Môi trường: Windows native, RTX 3090 24GB

## 1. Mục tiêu

Fine-tune model **Qwen3-0.6B** bằng Unsloth (QLoRA 4-bit) để:
- Giảm refusal (fine-tune trên dataset hội thoại đa chủ đề)
- Giữ/duy trì khả năng tiếng Việt

## 2. Môi trường & Cài đặt

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| Python | 3.11.9 | Unsloth không hỗ trợ 3.13 (bản Windows Store) |
| torch | 2.6.0+cu124 | CUDA 12.4, RTX 3090 (sm_86) |
| unsloth | 2026.8.12 | Cài qua git + `[windows]` extra |
| triton-windows | 3.2.0.post21 | BẮT BUỘC khớp torch 2.6 (không dùng 3.7.x) |
| xformers | 0.0.29.post3 | Khớp torch 2.6; bản 0.0.35 gây lỗi DLL |
| torchao | 0.13.0 | Bản 0.18 yêu cầu torch 2.7+ |
| llama-cpp-python | 0.3.14 | Build CPU với VS2022 Build Tools (export GGUF) |

Venv: `D:\ScriptPython\Training\.unsloth-env`

### Lỗi đã gặp và cách xử lý
1. **`aoti_torch_dtype_bool` DLL error** → xformers 0.0.35 không tương thích torch 2.6 → hạ về `0.0.29.post3`.
2. **`AttrsDescriptor` import error** → triton-windows 3.7 không khớp torch 2.6 → hạ về `3.2.0.post21`.
3. **torchao `register_constant` error** → torchao 0.18 cần torch 2.7+ → hạ về `0.13.0`.
4. **GGUF không train được** → model `radenadri/Qwen3.5-0.8B-...` (kiến trúc hybrid SSM `qwen35`) không được Unsloth hỗ trợ. Chuyển sang **Qwen3-0.6B** (dense, có sẵn safetensors).

## 3. Dataset

- **FineTome-100k** (`mlabonne/FineTome-100k`): 100k hội thoại tiếng Anh, chuyên giảm refusal. Lấy **12.000 mẫu**.
- **Vietnamese-Multi-turn-Chat-Alpaca** (`5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`): 12.697 hội thoại tiếng Việt. Lấy **12.000 mẫu**.
- Trộn 50/50 + shuffle seed 42 → **24.000 mẫu** tổng.

Lý do giới hạn 12k/loại: 550 bước × 8 mẫu/bước = ~4400 mẫu được thấy. Nếu trộn cả 100k EN thì tiếng Việt chỉ chiếm ~5%, model sẽ quên tiếng Việt (từng trả lời bằng tiếng Thái).

## 4. Tham số fine-tune

| Tham số | Giá trị |
|---|---|
| LoRA r / alpha | 16 / 16 |
| Target modules | q,k,v,o,gate,up,down_proj |
| Batch / grad accum | 2 / 4 (=8 mẫu/bước) |
| max_steps | 550 |
| learning_rate | 2e-4 (linear warmup 5) |
| fp16/bf16 | bf16 |
| Optimizer | adamw_8bit |
| max_seq_length | 2048 |
| Thời gian | ~10 phút |
| **Final train loss** | **1.015** |

## 5. Grid Search tham số generation (32 combo)

Chạy `grid_search.py` (4 temp × 4 repeat_penalty × 2 top_p, 4 prompt tiếng Việt, đo tỷ lệ lặp từ).

**Combo tốt nhất:**
- **TEMP=0.7 | REP_PEN=1.1 | TOP_P=0.9** → avg_repeat=0.0019
- TEMP=0.85 | REP_PEN=1.1 | TOP_P=0.95 → 0.0028
- TEMP=1.0 cho điểm 0.0000 nhưng **lạc đề**, không dùng

**Kết luận:** temp 0.7–0.85 là cân bằng tốt nhất. Temp 1.0 dễ lạc đề, temp 0.6 trả lời cụt.

## 6. Tham số Modelfile (Ollama) - ĐÃ ÁP DỤNG

```
PARAMETER repeat_penalty 1.1
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

## 7. Sản phẩm & Đường dẫn

| Sản phẩm | Đường dẫn | Dung lượng |
|---|---|---|
| Model base | `models/Qwen3-0.6B` | ~1.4 GB |
| LoRA adapter | `output_lora/` | checkpoint 550 |
| Model merged 16-bit | `output_lora_merged/` | 1.4 GB |
| GGUF Q4_K_M | `gguf_out_gguf/output_lora_merged.Q4_K_M.gguf` | 461.8 MB |
| Model Ollama | `qwen3-0.6b-finetuned` | 484 MB |
| Kết quả grid search | `grid_search_results.txt` | |

## 8. Scripts

| Script | Chức năng |
|---|---|
| `finetune_qwen3.py` | Fine-tune QLoRA (đổi `max_steps` nếu muốn train lại) |
| `run_finetune.bat` | Chạy train tiện lợi |
| `test_model.py` | Test model merged 16-bit |
| `export_gguf.py` | Xuất GGUF Q4_K_M |
| `grid_search.py` | Grid search tham số generation |
| `Modelfile` | Định nghĩa model Ollama |

## 9. Lưu ý

- Model 0.6B vốn bị giới hạn về độ sâu suy luận. Grid search chỉ tối ưu tham số, không làm model "thông minh hơn".
- Muốn giảm refusal mạnh hơn → tăng `max_steps` lên 1000–3000 (dataset hiện tại đủ 24k mẫu cho ~3000 bước).
- Tokenizer báo warning regex Mistral — không ảnh hưởng hoạt động.
