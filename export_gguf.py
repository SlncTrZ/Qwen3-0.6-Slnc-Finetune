import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "D:/ScriptPython/Training/output_lora_merged",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

model.save_pretrained_gguf(
    "D:/ScriptPython/Training/gguf_out",
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF export done.")
