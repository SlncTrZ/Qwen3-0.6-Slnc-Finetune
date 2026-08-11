import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "D:/ScriptPython/Training/output_lora_merged",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

tests = [
    "What is 2+2?",
    "Write a short haiku about the sea.",
]

for q in tests:
    messages = [{"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)
    print("=" * 50)
    print("Q:", q)
    print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
