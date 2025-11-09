#Đoạn này là để load các thành phần cần thiết như là mô hình embedding, kết nối Pinecone, mô hình Gemini, và khởi tạo các thành phần Retrieval và Generation.
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from core.Retrieval import Retrieval
from core.Generator import Generator

from utils import get_bm25_vocabulary, save_chunks_to_json
from config.setting import settings

import json
import os

#Hàm dùng để load các thành phần cần thiết để triên khai dự án
def load_components():
    print("[INFO] Loading embedding model and BM25 vocabulary...")
    embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
    bm25, vocabulary = get_bm25_vocabulary()

    print("[INFO] Connecting to Pinecone...")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    dense_index = pc.Index(host=settings.HOST_DENSE)
    sparse_index = pc.Index(host=settings.HOST_SPARSE)

    print("[INFO] Connecting to Gemini Model...")
    rewrite_model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
    router_model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
    generator_model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

    print("[INFO] Initializing Retrieval and Generation components...")
    retriever = Retrieval(pc, dense_index, sparse_index, embedding_model, bm25, vocabulary)
    generator = Generator(generator_model)

    print("[INFO] Loading knowlege info...")
    

    return rewrite_model, router_model, retriever, generator

#Tạo user trống với 3 môn, mỗi môn sẽ có 3 cấp độ trong đó
def default_user(user_id: str):
    levels = ["beginner", "exam", "advanced"]
    subjects = ["lich-su-dang", "tu-tuong-ho-chi-minh", "triet-hoc"]

    return {
        "_id": user_id,
        "name": "",
        "last_guiding": {
            "subject": "None",
            "level": "None"
        },
        "chat_history": [],
        "subjects": {
            subject: {
                level: {
                    "progress_concepts": []
                }
                for level in levels
            }
            for subject in subjects
        }
    }

def load_user(user_id: str, file_path: str = "./users/users.json"):
    """Tải user theo ID từ file, nếu chưa có thì tạo mới."""
    users = []

    # --- Bước 1: Đọc dữ liệu file ---
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read().strip()
                if data:  # tránh lỗi file rỗng
                    users = json.loads(data)
        except (json.JSONDecodeError, FileNotFoundError):
            users = []  # nếu lỗi đọc JSON, reset danh sách

    # --- Bước 2: Tìm user theo ID ---
    for user in users:
        if user.get("_id") == user_id:
            # Đảm bảo có đủ key (phòng lỗi khi format cũ)
            user.setdefault("last_guiding", {})
            user.setdefault("chat_history", [])
            user.setdefault("subjects", default_user(user_id)["subjects"])
            return user

    # --- Bước 3: Nếu không có thì tạo mới ---
    new_user = default_user(user_id)
    users.append(new_user)

    # --- Bước 4: Lưu lại file ---
    save_chunks_to_json(users, file_path)

    return new_user