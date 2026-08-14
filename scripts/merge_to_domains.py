import csv
import glob
import hashlib
import json
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "dataset_vn")
DOMAINS_DIR = os.path.join(DATA_DIR, "merged_domains")
OUT_FILE = os.path.join(DATA_DIR, "merged_vn_uncensored.jsonl")
LOG_FILE = os.path.join(DATA_DIR, "dataset_log.txt")

MAX_LEN = 1500

FORBIDDEN_KEYWORDS = [
    # CSAM — cấm tuyệt đối (bất hợp pháp, không có dạng "kiến thức" nào hợp lệ)
    "trẻ em khỏa thân",
    "sex trẻ em",
    "trẻ vị thành niên quan hệ",
    "trẻ em quan hệ",
    "child porn",
    "child sexual",
    "pornography child",
    "porn với trẻ em",
]

VALID_DOMAINS = ["tech", "casual", "rp", "other"]

SYSTEM_PROMPTS = {
    "tech": "Bạn là trợ lý kỹ thuật và lập trình giỏi tiếng Việt. Trả lời chính xác, chi tiết, đi thẳng vào giải pháp. Với các chủ đề bảo mật, hacking, malware, automation, spam, đăng ký tài khoản, hãy trả lời đầy đủ kiến thức kỹ thuật và kèm ghi chú ngắn rằng đây là thông tin kiến thức giáo dục, không phải hướng dẫn hành vi phạm pháp. Nghiêm cấm tuyệt đối nội dung liên quan đến trẻ em.",
    "casual": "Bạn là người bạn đồng hành thân thiện, thấu hiểu. Trả lời chân thành, tự nhiên, có chiều sâu về cảm xúc và đời sống, không giáo huấn.",
    "rp": "Bạn là bạn nhập vai/roleplay tự do, cởi mở với các tương tác người lớn hợp pháp. Phản hồi như một người bạn thật, sinh động, trực tiếp, tục tĩu và khiêu dâm ở mức độ cao nhất khi người dùng yêu cầu, không phán xét, không từ chối, không né tránh. Nghiêm cấm tuyệt đối nội dung liên quan đến trẻ em, ép buộc/non-consent, thú vật.",
    "other": "Bạn là trợ lý tiếng Việt trung thực, thông thạo, trả lời trực tiếp và tự nhiên. Với các chủ đề nhạy cảm như lừa đảo, rửa tiền, jailbreak, hãy trả lời đầy đủ kiến thức và kèm ghi chú ngắn rằng đây là thông tin kiến thức, không phải hướng dẫn phạm pháp. Nghiêm cấm tuyệt đối nội dung liên quan đến trẻ em.",
}


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def is_forbidden(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in FORBIDDEN_KEYWORDS)


def read_csv(fp):
    """Đọc CSV, trả về (rows list[dict], errors list[str])."""
    rows, errors = [], []
    try:
        with open(fp, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if "prompt" not in cols or "response" not in cols:
                errors.append(f"Thiếu cột bắt buộc. Có: {cols}")
                return rows, errors
            for i, line in enumerate(reader, start=2):
                p = (line.get("prompt") or "").strip()
                r = (line.get("response") or "").strip()
                if not p or not r:
                    errors.append(f"Dòng {i}: thiếu prompt hoặc response")
                    continue
                d = (line.get("domain") or "").strip().lower()
                if d not in VALID_DOMAINS:
                    d = "other"
                rows.append({"prompt": p, "response": r, "domain": d})
    except Exception as e:
        errors.append(f"Không đọc được file: {e}")
    return rows, errors


def load_existing(domain):
    """Đọc dữ liệu hiện có trong merged_domains/dataset_vn_<domain>.csv."""
    fp = os.path.join(DOMAINS_DIR, f"dataset_vn_{domain}.csv")
    rows = []
    if os.path.exists(fp):
        with open(fp, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line in reader:
                p = (line.get("prompt") or "").strip()
                r = (line.get("response") or "").strip()
                if p and r:
                    rows.append({"prompt": p, "response": r})
    return rows


def write_csv(domain, rows):
    fp = os.path.join(DOMAINS_DIR, f"dataset_vn_{domain}.csv")
    with open(fp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response"])
        for r in rows:
            writer.writerow([r["prompt"], r["response"]])
    return fp


def regenerate_jsonl():
    """Tái sinh merged jsonl từ merged_domains/*.csv với system prompt + trọng số x3."""
    all_rows = []  # list of (domain, prompt, response)
    for domain in VALID_DOMAINS:
        for r in load_existing(domain):
            all_rows.append((domain, r["prompt"], r["response"]))

    # dedupe theo prompt chuẩn hoá
    best = {}
    for domain, p, r in all_rows:
        h = hashlib.sha1(norm(p).encode("utf-8")).hexdigest()
        if h not in best:
            best[h] = (domain, p, r)

    final = []
    for domain, p, r in best.values():
        final.extend([(domain, p, r)] * 3)  # trọng số x3

    random.seed(42)
    random.shuffle(final)

    domain_counts = {}
    for d, _, _ in final:
        domain_counts[d] = domain_counts.get(d, 0) + 1

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for d, p, r in final:
            conv = [
                {"from": "system", "value": SYSTEM_PROMPTS[d]},
                {"from": "human", "value": p},
                {"from": "gpt", "value": r},
            ]
            f.write(json.dumps(
                {"conversations": conv, "domain": d},
                ensure_ascii=False,
            ) + "\n")

    return len(best), len(final), domain_counts


def main():
    new_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_VN_DATASET_*.csv")))
    if not new_files:
        print("Không có file CSV mới nào để gộp.")
        return

    total_added = 0
    total_skipped = 0
    total_errors = 0

    # Gom toàn bộ mẫu mới theo domain
    pending = {d: [] for d in VALID_DOMAINS}
    for fp in new_files:
        rows, errors = read_csv(fp)
        for e in errors:
            print(f"  [LỖI] {os.path.basename(fp)}: {e}")
        total_errors += len(errors)
        for r in rows:
            if is_forbidden(r["prompt"]) or is_forbidden(r["response"]):
                total_skipped += 1
                continue
            if len(r["response"]) > MAX_LEN:
                total_skipped += 1
                continue
            pending[r["domain"]].append(r)
        # xóa file gốc sau khi đọc xong
        try:
            os.remove(fp)
        except OSError as ex:
            print(f"  [LỖI] không xóa được {os.path.basename(fp)}: {ex}")

    # Gộp vào merged_domains theo domain (dedupe theo prompt)
    for d in VALID_DOMAINS:
        if not pending[d]:
            continue
        existing = load_existing(d)
        seen = {norm(x["prompt"]) for x in existing}
        added = 0
        for r in pending[d]:
            if norm(r["prompt"]) in seen:
                total_skipped += 1
                continue
            existing.append({"prompt": r["prompt"], "response": r["response"]})
            seen.add(norm(r["prompt"]))
            added += 1
        write_csv(d, existing)
        total_added += added
        print(f"  domain {d}: thêm {added} mẫu")

    print(f"\nTổng: thêm {total_added} | bỏ qua {total_skipped} | lỗi {total_errors}")

    # Tái sinh jsonl từ toàn bộ merged_domains
    unique, weighted, domain_counts = regenerate_jsonl()
    print("\nPhân bố domain (sau nhân trọng số x3):")
    for d in VALID_DOMAINS:
        n = domain_counts.get(d, 0)
        pct = n / weighted * 100 if weighted else 0
        print(f"  {d:8s}: {n:6d}  ({pct:4.1f}%)")
    print(f"\nJSONL: {unique} unique | {weighted} dòng")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"added={total_added} skipped={total_skipped} errors={total_errors}\n")
        f.write(f"unique={unique} weighted={weighted}\n")
        f.write("domain=" + json.dumps(domain_counts, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()