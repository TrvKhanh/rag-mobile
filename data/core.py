import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------
# Chroma setup
# -------------------------
path_database = "./data.chromadb"
client = chromadb.PersistentClient(path_database)

embf = embedding_functions.SentenceTransformerEmbeddingFunction(
    "bkai-foundation-models/vietnamese-bi-encoder", trust_remote_code=True
)
collection = client.get_or_create_collection(
    name="production",
    embedding_function=embf
)

# -------------------------
# Text cleaning / splitting
# -------------------------
NOISE_PATTERNS = [
    r"GP số\s*\d+[/A-Z\-\s0-9]*",
    r"Địa chỉ[:\s].*",
    r"Chịu trách nhiệm.*",
    r"Xem trung tâm bảo hành",
]

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace(";", ". ")
    for pat in NOISE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 1000:  # truncate long chunks
        text = text[:1000].rsplit(" ", 1)[0] + "..."
    return text

def split_into_sentences(text: str):
    # Simple sentence split by dot, | or newline
    text = text.replace("|", ".")
    sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
    return sentences

def chunk_text(text: str, max_tokens=200):
    # Token estimate ~1 token = 4 characters
    approx_chunk_size = max_tokens * 4
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""
    for s in sentences:
        if len(current_chunk) + len(s) + 2 <= approx_chunk_size:
            current_chunk += (" " if current_chunk else "") + s
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = s
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# -------------------------
# Chunking logic
# -------------------------
TOPIC_FIELDS = [
    "man_hinh", "camera_sau", "camera_truoc", "pin_sac",
    "he_dieu_hanh_cpu", "ket_noi", "thiet_ke_trong_luong",
    "tien_ich", "thoi_diem_ra_mat", "bao_hanh", "khuyen_mai"
]

def add_product_to_collection(product: dict, index: int, collection):
    total_chunks = 0
    base_metadata = {
        "title": product.get("ten_san_pham"),
        "price": str(product.get("gia")),
        "url": product.get("url"),
        "image_url": product.get("hinh_anh")
    }
    base_metadata = {k: v for k, v in base_metadata.items() if v is not None and pd.notna(v)}

    # Specs chunking
    specs = product.get("specs", {})
    for field in TOPIC_FIELDS:
        content = specs.get(field)
        if content:
            content = clean_text(content)
            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                doc_id = f"{index}_{field}_{i}"
                collection.add(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[{**base_metadata, "topic": field}]
                )
                total_chunks += 1

    # Chi_nhanh chunking
    branches = product.get("chi_nhanh", [])
    for i, branch in enumerate(branches):
        branch_text = clean_text(branch)
        if branch_text:
            doc_id = f"{index}_branch_{i}"
            collection.add(
                ids=[doc_id],
                documents=[branch_text],
                metadatas=[{**base_metadata, "topic": "chi_nhanh"}]
            )
            total_chunks += 1

    return total_chunks

# -------------------------
# Main pipeline
# -------------------------
def main(csv_path: str):
    # Clear collection
    try:
        client.delete_collection(name="production")
    except:
        pass
    collection = client.get_or_create_collection(
        name="production",
        embedding_function=embf
    )

    df = pd.read_csv(csv_path)
    df.dropna(subset=["ten_san_pham"], inplace=True)

    total_chunks = 0
    for idx, row in df.iterrows():
        product = row.to_dict()
        total_chunks += add_product_to_collection(product, idx, collection)

    print(f"Total chunks added: {total_chunks}")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    csv_file = "/home/big/Projects/rag-mobile/data/hoangha_normalized.csv"
    main(csv_file)
