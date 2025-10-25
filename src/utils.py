import numpy as np
import json
import os
from rank_bm25 import BM25Okapi
from prompts.prompts import summarize_instructor

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

def get_bm25_vocabulary():

    triet_hoc_path = r"./data/TrietHoc/chunks/TrietHoc_Raw.json"
    lich_su_dang_path = r"./data/LichSuDang/chunks/Lich_Su_Dang_Raw.json"
    tu_tuong_hcm_path = r"./data/TuTuongHoChiMinh/chunks/TT_HCM_Raw.json"

    raw_chunk = load_chunks_from_json(lich_su_dang_path) + load_chunks_from_json(triet_hoc_path) + load_chunks_from_json(tu_tuong_hcm_path)

    # Tạo corpus
    corpus_texts = [chunk["content"] for chunk in raw_chunk]
    tokenized_corpus = [bm25_tokenize(text) for text in corpus_texts]

    bm25 = BM25Okapi(tokenized_corpus)
    vocabulary = list(bm25.idf.keys())

    return bm25, vocabulary

def summarize_answer(answer, summarize_model):
    chain = summarize_instructor | summarize_model
    response = chain.invoke({"query": answer}).content.strip().lower()
    return response