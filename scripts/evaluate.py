# -*- coding: utf-8 -*-
import sys
import json
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-0.6b-finetuned"

TESTS = [
    ("Cơ bản", "1 + 1 = ?"),
    ("Tiếng Việt ngắn", "Viết một câu giới thiệu bản thân."),
    ("Đọc hiểu ngữ cảnh", "Tôi có 3 quả táo, mua thêm 2 quả, ăn 1 quả. Hỏi còn mấy quả?"),
    ("Từ vựng yêu cầu", "Nêu 3 loại trái cây nhiệt đới của Việt Nam."),
    ("Định dạng có cấu trúc", "Liệt kê 3 bước để nấu mì gói, mỗi bước một dòng."),
    ("Dịch thuật", "Dịch sang tiếng Anh: 'Hôm nay trời đẹp, tôi đi dạo công viên.'"),
    ("Sáng tạo ngắn", "Đặt tên một chú mèo và nói lý do."),
    ("Giải thích khái niệm", "Lãi kép là gì? Giải thích đơn giản."),
]

for name, prompt in TESTS:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "repeat_penalty": 1.1, "top_p": 0.9, "num_predict": 120},
    }
    try:
        r = requests.post(URL, json=payload, timeout=120)
        out = r.json().get("response", "")
    except Exception as e:
        out = f"ERROR: {e}"
    print("=" * 60)
    print(f"[{name}] {prompt}")
    print("-" * 60)
    print(out.strip())
    print()
