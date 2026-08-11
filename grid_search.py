# -*- coding: utf-8 -*-
"""
Grid Search tham số generation cho model qwen3-0.6b-finetuned qua Ollama API local.
Chạy: .unsloth-env\Scripts\python.exe grid_search.py
"""
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-0.6b-finetuned"

# Lưới tham số cần quét
TEMPERATURES = [0.6, 0.7, 0.85, 1.0]
REPETITION_PENALTIES = [1.1, 1.15, 1.2, 1.25]
TOP_P = [0.9, 0.95]

# Prompt test (có cả câu thông thường + câu dễ khiến model lặp từ)
TEST_PROMPTS = [
    "Viết một đoạn văn phân tích về ranh giới giữa con người và máy móc.",
    "Kể về một ngày mùa thu ở làng quê Việt Nam, khoảng 100 từ.",
    "Giải thích ngắn gọn hiện tượng nắng nóng đô thị (urban heat island).",
    "Viết một lời chúc mừng năm mới dành cho gia đình, không quá 50 từ.",
]

MAX_NEW_TOKENS = 150
OUTPUT_FILE = "grid_search_results.txt"


def generate(prompt: str, temperature: float, repeat_penalty: float, top_p: float) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "top_p": top_p,
            "num_predict": MAX_NEW_TOKENS,
        },
    }
    resp = requests.post(URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")


def word_repetition_ratio(text: str) -> float:
    """Tỷ lệ token lặp lại gần nhau - càng cao càng ngớ ngẩn."""
    words = [w.strip(".,!?;:()") for w in text.split()]
    words = [w for w in words if w]
    if len(words) < 4:
        return 0.0
    repeats = sum(1 for i in range(1, len(words)) if words[i].lower() == words[i - 1].lower())
    return repeats / len(words)


def main():
    lines = []
    header = "=" * 70
    lines.append(header)
    lines.append(f"GRID SEARCH | model={MODEL} | prompts={len(TEST_PROMPTS)}")
    lines.append(f"combos={len(TEMPERATURES) * len(REPETITION_PENALTIES) * len(TOP_P)}")
    lines.append(header)

    results = []
    total = len(TEMPERATURES) * len(REPETITION_PENALTIES) * len(TOP_P)
    count = 0

    for temp in TEMPERATURES:
        for rep_pen in REPETITION_PENALTIES:
            for top_p in TOP_P:
                count += 1
                print(f"[{count}/{total}] TEMP={temp} | REP_PEN={rep_pen} | TOP_P={top_p}")
                lines.append("\n" + header)
                lines.append(f"TEMP={temp} | REP_PEN={rep_pen} | TOP_P={top_p}")
                lines.append(header)
                combo_stats = []
                for i, prompt in enumerate(TEST_PROMPTS, 1):
                    try:
                        out = generate(prompt, temp, rep_pen, top_p)
                    except Exception as e:
                        out = f"ERROR: {e}"
                    ratio = word_repetition_ratio(out) if not out.startswith("ERROR") else 0.0
                    combo_stats.append(ratio)
                    lines.append(f"\n--- Prompt {i}: {prompt[:60]}...")
                    lines.append(out.strip())
                    lines.append(f"[repeat_ratio={ratio:.3f}]")
                    time.sleep(0.5)
                results.append({
                    "temp": temp, "rep_pen": rep_pen, "top_p": top_p,
                    "avg_repeat_ratio": sum(combo_stats) / len(combo_stats),
                })

    # Bảng xếp hạng theo tỷ lệ lặp từ (thấp = tốt)
    lines.append("\n" + "=" * 70)
    lines.append("XẾP HẠNG THEO TỶ LỆ LẶP TỪ TRUNG BÌNH (THẤP = TỐT)")
    lines.append("=" * 70)
    for r in sorted(results, key=lambda x: x["avg_repeat_ratio"]):
        lines.append(f"  TEMP={r['temp']:<5} REP_PEN={r['rep_pen']:<5} TOP_P={r['top_p']:<5} avg_repeat={r['avg_repeat_ratio']:.4f}")

    report = "\n".join(lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print("\nDone! Kết quả lưu tại:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
