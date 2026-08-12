# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "D:/ScriptPython/Training/models/Qwen3-0.6B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

TESTS = [
    "1 + 1 = ?",
    "Nêu 3 loại trái cây nhiệt đới của Việt Nam.",
    "Tôi có 3 quả táo, mua thêm 2 quả, ăn 1 quả. Hỏi còn mấy quả?",
]

for prompt in TESTS:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=120, temperature=0.7, top_p=0.9)
    print("=" * 50)
    print("Q:", prompt)
    print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
