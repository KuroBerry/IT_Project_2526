from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Import logic cũ của bạn (Giữ nguyên không sửa gì cả) ---
from server.src.core.Retrieval import Retrieval
from server.src.core.Generator import Generator
from server.src.core.ChatManager import ChatManager
from server.src.core.loader import load_components
from server.users.user_manager import load_user

# 1. Khởi tạo App
app = FastAPI()

# --- PHẦN 1: SETUP (Chạy 1 lần duy nhất khi khởi động Server) ---
# Trong file cũ, phần này nằm trước while True
print("⏳ Đang load Models... Vui lòng đợi...")

# Load components 1 lần thôi, để sẵn trong RAM dùng chung cho mọi user
rewrite_model, router_model, retriever, generator = load_components()

print("✅ Server đã sẵn sàng!")

# --- PHẦN 2: ĐỊNH NGHĨA DỮ LIỆU INPUT (Thay cho input()) ---
# Quy định user phải gửi gì lên
class ChatRequest(BaseModel):
    user_id: str = "U001" # Mặc định là U001 nếu không gửi
    query: str
    top_k: int = 10

# --- PHẦN 3: XỬ LÝ REQUEST (Thay cho vòng lặp while True) ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Hàm này thay thế cho đoạn: query = input(...)
    """
    
    # 1. Load user (Giả lập logic cũ của bạn)
    # Lưu ý: Load file json mỗi lần request có thể chậm, sau này nên optimize

    user = await load_user(request.user_id)


    # 2. Gọi Logic (Giống hệt file test)
    # Thay vì query lấy từ input(), ta lấy từ request.query
    chat_manager = ChatManager(user, rewrite_model, router_model, retriever, generator)
    result, history = await chat_manager.handle_query(request.query, request.top_k)

    # 3. Trả về kết quả (Thay thế cho print)
    # Trả về JSON để Frontend/Mobile App đọc được
    return {
        "status": "success",
        "answer": result,    # Câu trả lời của AI
        "user_id": request.user_id
    }

# Để chạy: uvicorn main:app --reload