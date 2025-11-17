# csv_to_clean_json.py
import pandas as pd
import json
import re

# Các trường specs thực sự
SPEC_FIELDS = [
    "man_hinh", "camera_sau", "camera_truoc", "pin_sac",
    "he_dieu_hanh_cpu", "ket_noi", "thiet_ke_trong_luong",
    "tien_ich", "thong_tin_chung", "bao_hanh", "khuyen_mai"
]

# Các pattern để loại bỏ text "nhiễu"
NOISE_PATTERNS = [
    r"GP số\s*\d+[/A-Z\-\s0-9]*",
    r"Địa chỉ[:\s].*",
    r"Chịu trách nhiệm.*",
    r"mình tham khảo",
    r"Xem trung tâm bảo hành"
]

# Chunk long text > max_len
def chunk_text(text, max_len=1000):
    text = text.strip()
    if len(text) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        # tránh cắt nửa chữ
        if end < len(text):
            end = text.rfind(" ", start, end)
            if end == -1: end = start + max_len
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# Clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    t = str(text).replace(";", ". ")
    t = re.sub(r"\s+", " ", t).strip()
    for pat in NOISE_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    return t

# Normalize price
def normalize_price(val):
    if pd.isna(val):
        return None
    s = str(val)
    digits = re.sub(r"[^0-9]", "", s)
    return int(digits) if digits else None

# Split list field
def split_list_field(val):
    if pd.isna(val):
        return []
    parts = re.split(r"[;,/]+", str(val))
    return [p.strip() for p in parts if p.strip()]

def normalize_row(row):
    obj = {}
    obj["ten_san_pham"] = str(row.get("ten_san_pham", "")).strip()
    obj["gia"] = normalize_price(row.get("gia"))
    obj["hinh_anh"] = row.get("hinh_anh") or row.get("url_anh") or ""
    obj["url"] = row.get("url", "")
    obj["thuong_hieu"] = row.get("thuong_hieu", "")
    obj["mau_sac"] = split_list_field(row.get("mau_sac", ""))
    obj["tinh_trang"] = clean_text(row.get("tinh_trang", "Địa chỉ còn hàng"))

    # Specs
    specs = {}
    for f in SPEC_FIELDS:
        value = row.get(f)
        if pd.notna(value) and str(value).strip():
            text = clean_text(value)
            # chunk nếu quá dài
            chunks = chunk_text(text)
            # nếu nhiều chunk, join bằng \n---chunk---\n
            specs[f] = "\n---chunk---\n".join(chunks)
        else:
            specs[f] = ""
    obj["specs"] = specs

    # chi_nhanh
    branches = split_list_field(row.get("chi_nhanh", ""))
    obj["chi_nhanh"] = [b for b in branches if b]

    return obj

def main(csv_path, out_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["ten_san_pham"], inplace=True)
    result = [normalize_row(r) for _, r in df.iterrows()]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote normalized JSON to {out_path}, total products: {len(result)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON")
    args = parser.parse_args()
    main(args.csv, args.out)
