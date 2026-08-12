import csv
import glob
import hashlib
import json
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_vn")
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
    # Hành vi phạm tội nhắm nạn nhân thực (giữ cấm; kiến thức bảo mật giáo dục vẫn được phép)
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

MIN_LEN = 20
MAX_LEN = 1500


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
                rows.append({"prompt": p, "response": r})
    except Exception as e:
        errors.append(f"Không đọc được file: {e}")
    return rows, errors, warnings


def is_forbidden(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in FORBIDDEN_KEYWORDS)


def main():
    custom_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_VN_DATASET_*.csv")))
    custom_files += sorted(glob.glob(os.path.join(DATA_DIR, "custom_*.csv")))
    custom_files = sorted(set(custom_files))
    aux_files = sorted(glob.glob(os.path.join(DATA_DIR, "aux_*.csv")))
    allow_files = sorted(glob.glob(os.path.join(DATA_DIR, "_manual_allowlist_*.csv")))

    all_rows = []  # list of (weight, row)
    total_errors = 0
    blocked = 0
    too_long = 0

    print("== Gộp file CUSTOM (trọng số x3) ==")
    for fp in custom_files:
        rows, errors, _ = validate_file(fp)
        for e in errors:
            print(f"  [LỖI] {os.path.basename(fp)}: {e}")
        total_errors += len(errors)
        kept = 0
        for r in rows:
            if is_forbidden(r["prompt"]) or is_forbidden(r["response"]):
                blocked += 1
                continue
            if len(r["response"]) < MIN_LEN or len(r["response"]) > MAX_LEN:
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
        for r in rows:
            if is_forbidden(r["prompt"]) or is_forbidden(r["response"]):
                blocked += 1
                continue
            if len(r["response"]) < MIN_LEN or len(r["response"]) > MAX_LEN:
                too_long += 1
                continue
            all_rows.append((1, r))
            kept += 1
        print(f"  {os.path.basename(fp)}: {kept} mẫu giữ lại")

    print("\n== Gộp file ALLOWLIST (trọng số x3, bỏ qua bộ lọc cấm) ==")
    for fp in allow_files:
        rows, errors, _ = validate_file(fp)
        for e in errors:
            print(f"  [LỖI] {os.path.basename(fp)}: {e}")
        total_errors += len(errors)
        kept = 0
        for r in rows:
            if len(r["response"]) < MIN_LEN or len(r["response"]) > MAX_LEN:
                too_long += 1
                continue
            all_rows.append((3, r))
            kept += 1
        print(f"  {os.path.basename(fp)}: {kept} mẫu giữ lại")

    print(
        f"\nTổng lỗi: {total_errors} | Bị chặn (cấm): {blocked} | Quá ngắn/dài: {too_long}"
    )

    # Dedupe theo prompt chuẩn hoá (giữ bản có trọng số cao nhất)
    best = {}
    for weight, r in all_rows:
        h = hashlib.sha1(norm(r["prompt"]).encode("utf-8")).hexdigest()
        if h not in best or weight > best[h][0]:
            best[h] = (weight, r)

    final = []
    for weight, r in best.values():
        final.extend([r] * weight)

    random.seed(42)
    random.shuffle(final)

    print(
        f"\nSau dedupe: {len(best)} mẫu duy nhất | Sau nhân trọng số: {len(final)} dòng"
    )

    try:
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            for r in final:
                conv = [
                    {"from": "human", "value": r["prompt"]},
                    {"from": "gpt", "value": r["response"]},
                ]
                f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[LỖI] Không ghi được {OUT_FILE}: {e}")
        return

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"custom_files={len(custom_files)} aux_files={len(aux_files)}\n")
            f.write(f"allow_files={len(allow_files)}\n")
            f.write(f"unique={len(best)} weighted={len(final)}\n")
            f.write(
                f"errors={total_errors} blocked={blocked} len_filtered={too_long}\n"
            )
    except OSError as e:
        print(f"[LỖI] Không ghi được {LOG_FILE}: {e}")

    print(f"\nĐÃ GHI: {OUT_FILE} ({len(final)} dòng)  | Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
