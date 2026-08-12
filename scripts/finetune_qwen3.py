import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse

parser = argparse.ArgumentParser(description="Fine-tune QLoRA 4-bit bằng Unsloth")
parser.add_argument("--model", default="D:/ScriptPython/Training/models/Qwen3-0.6B",
                    help="Đường dẫn model base (mặc định: models/Qwen3-0.6B)")
parser.add_argument("--steps", type=int, default=500,
                    help="Số bước train max_steps (vd 200 / 500 / 5000)")
parser.add_argument("--dataset", default="D:/ScriptPython/Training/dataset_vn/merged_vn_uncensored.jsonl",
                    help="File dataset jsonl (mặc định: merged_vn_uncensored.jsonl)")
parser.add_argument("--output", default="D:/ScriptPython/Training/output_lora",
                    help="Thư mục output LoRA + merged")
args = parser.parse_args()

MODEL_PATH = args.model
OUTPUT_DIR = args.output
DATASET_FILE = args.dataset
MAX_STEPS = args.steps

MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

print("=== CẤU HÌNH FINETUNE ===")
print("Model :", MODEL_PATH)
print("Dataset:", DATASET_FILE)
print("Steps :", MAX_STEPS)
print("Output:", OUTPUT_DIR)
print("========================")

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
print("Total samples:", len(dataset))

def format_chat(examples):
    texts = []
    for conv in examples["conversations"]:
        chat = []
        for turn in conv:
            role = turn["from"]
            if role == "human":
                chat.append({"role": "user", "content": turn["value"]})
            elif role == "gpt":
                chat.append({"role": "assistant", "content": turn["value"]})
            else:
                chat.append({"role": "system", "content": turn["value"]})
        texts.append(tokenizer.apply_chat_template(chat, tokenize=False))
    return {"text": texts}

dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

trainer_stats = trainer.train()
print("Training done:", trainer_stats)

model.save_pretrained_merged(OUTPUT_DIR + "_merged", tokenizer, save_method="merged_16bit")
print("Merged 16-bit model saved to", OUTPUT_DIR + "_merged")
