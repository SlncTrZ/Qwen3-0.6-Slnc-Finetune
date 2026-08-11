# Qwen3-0.6-Slnc-Finetune

Fine-tune **Qwen3-0.6B** bằng Unsloth (QLoRA 4-bit) trên GPU RTX 3090.
Dataset EN+VI 50/50, xuất GGUF Q4_K_M cho Ollama.

Chi tiết quy trình: xem [record.md](record.md)

## Scripts

| Script | Chức năng |
|---|---|
| `finetune_qwen3.py` | Fine-tune QLoRA (EN+VI dataset) |
| `run_finetune.bat` | Chạy train tiện lợi |
| `test_model.py` | Test model merged 16-bit |
| `export_gguf.py` | Xuất GGUF Q4_K_M |
| `grid_search.py` | Grid search tham số generation |
| `Modelfile` | Định nghĩa model Ollama |

## Yêu cầu

- Python 3.11
- torch 2.6.0+cu124, triton-windows 3.2.0.post21, xformers 0.0.29.post3, torchao 0.13.0
- unsloth (bản mới nhất)
- llama-cpp-python 0.3.14 (cho export GGUF)

## Chạy train

```powershell
D:\ScriptPython\Training\run_finetune.bat
```

## Tham số tối ưu (Ollama Modelfile)

```
PARAMETER repeat_penalty 1.1
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```
