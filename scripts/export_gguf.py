import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse

parser = argparse.ArgumentParser(description="Export model merged sang GGUF Q4_K_M")
parser.add_argument("--model", default="D:/ScriptPython/Training/output_lora_merged",
                    help="Thư mục model merged 16-bit (mặc định: output_lora_merged)")
parser.add_argument("--out", default="D:/ScriptPython/Training/gguf_out_gguf",
                    help="Thư mục xuất GGUF (mặc định: gguf_out_gguf)")
args = parser.parse_args()

MODEL_PATH = args.model
OUT_DIR = args.out

print("Export GGUF:", MODEL_PATH, "->", OUT_DIR)

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

model.save_pretrained_gguf(
    OUT_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF export done.")
