import csv
import glob
import hashlib
import json
import os
import random
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "dataset_vn")
OUT_FILE = os.path.join(DATA_DIR, "merged_vn_uncensored.jsonl")
LOG_FILE = os.path.join(DATA_DIR, "dataset_log.txt")

FORBIDDEN_KEYWORDS = [
    # CSAM — cấm tuyệt đối, không bao giờ nới
    "trẻ em khỏa thân",
    "sex trẻ em",
    "trẻ vị thành niên quan hệ",
    "child porn",
    "cp ",
    # Vũ khí / chất nổ / ma túy / hóa chất cấm — cấm tuyệt đối
    "chế tạo bom",
    "làm bom",
    "c4 ",
    "thuốc nổ",
    "điều chế ma túy",
    "nấu ma túy",
    "methamphetamine",
    "sản xuất vũ khí",
    "chế tạo súng",
    "làm súng",
    "sí",
    "ma túy tổng hợp",
    # Tự hại
    "hướng dẫn tự sát",
    "cách tự tử",
    # Hành vi phạm tội nhắm nạn nhân thực
    "hack vào tài khoản",
    "crack tài khoản của",
    "bẻ khóa tài khoản của",
    "trộm tài khoản",
    "xâm nhập trái phép",
    "đánh cắp mật khẩu",
    "đánh cắp tài khoản",
    "hack vào wifi",
    "lừa đảo chiếm đoạt",
    "chiếm đoạt tài sản",
    "lừa tiền của",
    "rửa tiền",
    "phát tán mã độc",
    "tống tiền",
    "tấn công hệ thống của",
    "ddos website của",
]

MIN_LEN = 60
MAX_LEN = 1500

# Bonus đối với keyword "hack" — hack học thuật/bảo mật giáo dục vẫn được phép
ALLOWED_HACK_CONTEXT = [
    "phòng lab",
    "học",
    "giáo dục",
    "xâm nhập an toàn",
    "kiểm thử bảo mật",
    "pentest",
    "bug bounty",
    "pháp lý",
    "hợp pháp",
]

RP_KEYWORDS = [
    "sex", "ngực", "dương vật", "âm đạo", "thủ dâm", "oral", "địt", "chich",
    "phim sex", "khiêu dâm", "18+", "nhập vai", "tán tỉnh", "vuốt ve", "hôn",
    "bạn gái", "bạn trai", "chăn gối", "làm tình", "quan hệ tình dục", "cưỡi",
    "xuất tinh", "múp", "tử cung", "ngực đẹp", "chịch", "bú", "liếm", "mông",
    "móc", "bắn", "rên", "ham muốn", "hưng phấn", "mát-xa", "dâm", "cặc",
    "lồn", "sex chat", "ảnh nóng", "video nóng", "gạ tình", "ân ái", "sướng",
]

TECH_KEYWORDS = [
    "python", "javascript", "java", "html", "css", "sql", "database", "api",
    "server", "code", "lập trình", "thuật toán", "bug", "debug", "git", "docker",
    "linux", "windows", "android", "iphone", "máy tính", "phần mềm", "ứng dụng",
    "website", "app", "ai", "machine learning", "deep learning", "neural", "gpt",
    "gpu", "cpu", "ram", "hack", "bảo mật", "mã độc", "virus", "wifi", "password",
    "mật khẩu", "script", "config", "chạy lệnh", "hàm", "biến", "oop", "framework",
    "excel", "word", "powerpoint", "kỹ thuật số", "robot", "automation", "chrome",
    "firefox", "browser", "cài đặt", "phần cứng", "mạng máy tính", "công nghệ",
]

CASUAL_KEYWORDS = [
    "tâm lý", "cảm xúc", "lo âu", "trầm cảm", "stress", "mệt mỏi", "buồn",
    "vui", "gia đình", "bố mẹ", "con cái", "hôn nhân", "ly hôn", "chia tay",
    "yêu đương", "tỏ tình", "sức khỏe", "bệnh", "tập thể dục", "ngủ", "ăn uống",
    "công việc", "đồng nghiệp", "bạn bè", "học tập", "học sinh", "sinh viên",
    "áp lực", "giấc mơ", "sở thích", "du lịch", "cuộc sống", "đời thường",
    "tự tin", "mặc cảm", "say rượu", "uống", "lối sống", "thói quen", "mục tiêu",
]

SYSTEM_PROMPTS = {
    "tech": "Bạn là trợ lý kỹ thuật và lập trình giỏi tiếng Việt. Trả lời chính xác, chi tiết, đi thẳng vào giải pháp. Với các chủ đề bảo mật/kiến thức nhạy cảm, trả lời ở mức độ giáo dục và hợp pháp.",
    "casual": "Bạn là người bạn đồng hành thân thiện, thấu hiểu. Trả lời chân thành, tự nhiên, có chiều sâu về cảm xúc và đời sống, không giáo huấn.",
    "rp": "Bạn là bạn nhập vai/roleplay tự do, cởi mở với các tương tác người lớn hợp pháp. Phản hồi như một người bạn thật, sinh động, trực tiếp, không phán xét, không từ chối. Nghiêm cấm nội dung liên quan đến trẻ em.",
    "other": "Bạn là trợ lý tiếng Việt trung thực, thông thạo, trả lời trực tiếp và tự nhiên.",
}

VALID_DOMAINS = set(SYSTEM_PROMPTS)


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def validate_file(fp):
    """Đọc 1 file CSV, trả về (rows, errors, warnings)."""
    rows = []
    errors = []
    warnings = []
    try:
        with open(fp, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if "prompt" not in cols or "response" not in cols:
                errors.append(f"Thiếu cột bắt buộc. Có: {cols}")
                return rows, errors, warnings
            for i, line in enumerate(reader, start=2):
                p = (line.get("prompt") or "").strip()
                r = (line.get("response") or "").strip()
                if not p or not r:
                    errors.append(f"Dòng {i}: thiếu prompt hoặc response")
                    continue
                d = (line.get("domain") or "").strip().lower()
                if d not in VALID_DOMAINS:
                    d = ""
                rows.append({"prompt": p, "response": r, "domain": d})
    except Exception as e:
        errors.append(f"Không đọc được file: {e}")
    return rows, errors, warnings


def is_forbidden(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in FORBIDDEN_KEYWORDS)


def classify_domain(text: str, hint: str = "") -> str:
    t = " " + hint.lower() + " " + text.lower() + " "
    if "tech" in hint.lower() or t.strip().startswith("tech "):
        return "tech"
    if "casual" in hint.lower() or t.strip().startswith("casual "):
        return "casual"
    if "rp" in hint.lower():
        return "rp"
    rp = any(k in t for k in RP_KEYWORDS)
    tech = any(k in t for k in TECH_KEYWORDS)
    casual = any(k in t for k in CASUAL_KEYWORDS)
    if rp:
        return "rp"
    if tech and not casual:
        return "tech"
    if casual:
        return "casual"
    if tech:
        return "tech"
    return "other"


# Bonus: nhưng "hack" dạng giáo dục trong tech — bỏ qua block từ khoá hack
def hack_is_allowed(text: str) -> bool:
    t = text.lower()
    if "hack vào" in t or "crack tài khoản" in t or "trộm tài khoản" in t:
        return False
    for ctx in ALLOWED_HACK_CONTEXT:
        if ctx in t:
            return True
    return False


def main():
    custom_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_VN_DATASET_*.csv")))
    custom_files += sorted(glob.glob(os.path.join(DATA_DIR, "custom_*.csv")))
    custom_files = sorted(set(custom_files))
    aux_files = sorted(glob.glob(os.path.join(DATA_DIR, "aux_*.csv")))

    all_rows = []  # list of (weight, row)
    total_errors = 0
    blocked = 0
    too_short = 0
    too_long = 0

    print("== Gộp file CUSTOM (trọng số x3) ==")
    for fp in custom_files:
        rows, errors, _ = validate_file(fp)
        for e in errors:
            print(f"  [LỖI] {os.path.basename(fp)}: {e}")
        total_errors += len(errors)
        kept = 0
        # hint từ tên file (hậu tố _domain) nếu có
        hint = os.path.basename(fp)
        for r in rows:
            if r["domain"] not in VALID_DOMAINS:
                r["domain"] = classify_domain(r["prompt"] + " " + r["response"], hint)
            full = r["prompt"] + " " + r["response"]
            if is_forbidden(full) and not hack_is_allowed(full):
                blocked += 1
                continue
            if len(r["response"]) < MIN_LEN:
                too_short += 1
                continue
            if len(r["response"]) > MAX_LEN:
                too_long += 1
                continue
            all_rows.append((3, r))
            kept += 1
        print(f"  {os.path.basename(fp)}: {kept} mẫu giữ lại")

    print("\n== Gộp file AUX (trọng số x1) ==")
    for fp in aux_files:
        rows, errors, _ = validate_file(fp)
        for e in errors:
            print(f"  [LỖI] {os.path.basename(fp)}: {e}")
        total_errors += len(errors)
        kept = 0
        hint = os.path.basename(fp)
        for r in rows:
            if r["domain"] not in VALID_DOMAINS:
                r["domain"] = classify_domain(r["prompt"] + " " + r["response"], hint)
            full = r["prompt"] + " " + r["response"]
            if is_forbidden(full) and not hack_is_allowed(full):
                blocked += 1
                continue
            if len(r["response"]) < MIN_LEN:
                too_short += 1
                continue
            if len(r["response"]) > MAX_LEN:
                too_long += 1
                continue
            all_rows.append((1, r))
            kept += 1
        print(f"  {os.path.basename(fp)}: {kept} mẫu giữ lại")

    print(
        f"\nTổng lỗi: {total_errors} | Bị chặn (cấm): {blocked} "
        f"| Quá ngắn(<{MIN_LEN}): {too_short} | Quá dài(>{MAX_LEN}): {too_long}"
    )

    # Dedupe theo prompt chuẩn hoá (giữ bản có trọng số cao nhất)
    best = {}
    for weight, r in all_rows:
        h = hashlib.sha1(norm(r["prompt"]).encode("utf-8")).hexdigest()
        if h not in best or weight > best[h][0]:
            best[h] = (weight, r)

    final = []
    for weight, r in best.values():
        for _ in range(weight):
            final.append(r)

    random.seed(42)
    random.shuffle(final)

    print(
        f"\nSau dedupe: {len(best)} mẫu duy nhất | Sau nhân trọng số: {len(final)} dòng"
    )

    # Thống kê domain + xuất file per-domain
    domain_counts = {}
    for r in final:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1

    print("\nPhân bố domain (sau nhân trọng số):")
    for d in sorted(VALID_DOMAINS):
        n = domain_counts.get(d, 0)
        pct = n / len(final) * 100 if final else 0
        print(f"  {d:8s}: {n:6d}  ({pct:4.1f}%)")
    n_other = domain_counts.get("other", 0)
    n_rp = domain_counts.get("rp", 0)

    try:
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            for r in final:
                system = SYSTEM_PROMPTS[r["domain"]]
                conv = [
                    {"from": "system", "value": system},
                    {"from": "human", "value": r["prompt"]},
                    {"from": "gpt", "value": r["response"]},
                ]
                f.write(json.dumps(
                    {"conversations": conv, "domain": r["domain"]},
                    ensure_ascii=False,
                ) + "\n")
    except OSError as e:
        print(f"[LỖI] Không ghi được {OUT_FILE}: {e}")
        return

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"custom_files={len(custom_files)} aux_files={len(aux_files)}\n")
            f.write(f"unique={len(best)} weighted={len(final)}\n")
            f.write(
                f"errors={total_errors} blocked={blocked} "
                f"too_short={too_short} too_long={too_long}\n"
            )
            f.write("domain=" + json.dumps(domain_counts, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[LỖI] Không ghi được {LOG_FILE}: {e}")

    print(
        f"\nĐÃ GHI: {OUT_FILE} ({len(final)} dòng)  | Log: {LOG_FILE}"
    )


if __name__ == "__main__":
    main()