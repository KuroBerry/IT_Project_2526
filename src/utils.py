import numpy as np
import json
import os

def bm25_tokenize(text):
    return text.lower().split()

def text_to_sparse_vector_bm25(text, bm25, vocabulary):
    tokens = bm25_tokenize(text)
    vector = np.zeros(len(vocabulary))
    for i, word in enumerate(vocabulary):
        idf = bm25.idf.get(word, 0)
        tf = tokens.count(word)
        vector[i] = idf * tf
    indices = vector.nonzero()[0].tolist()
    values = vector[indices].tolist()
    return {"indices": indices, "values": values}


def load_chunks_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

def save_chunks_to_json(chunks, output_path):
    """
    Lưu danh sách các chunk (list of dict) ra file JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Tạo folder nếu chưa có

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu {len(chunks)} chunks vào: {output_path}")