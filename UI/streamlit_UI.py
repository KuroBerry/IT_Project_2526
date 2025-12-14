import streamlit as st
import requests
import time
import os

# --- CẤU HÌNH KẾT NỐI ---
# Đây là địa chỉ Backend FastAPI của bạn
BACKEND_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="EduRAG Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- SIDEBAR (CẤU HÌNH) ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    # Cho phép nhập User ID để test nhiều user khác nhau
    user_id = st.text_input("User ID", value="U000")
    # Cho phép chỉnh Top K
    top_k = st.slider("Top K (Context)", min_value=1, max_value=20, value=10)
    
    st.divider()
    st.info("Frontend đang chạy độc lập. Đảm bảo bạn đã bật Backend ở port 8000.")

# --- GIAO DIỆN CHÍNH ---
st.title("🤖 EduRAG - Trợ lý học tập")
st.caption(f"Đang kết nối với Backend tại: `{BACKEND_URL}`")

# 1. Khởi tạo Session State (Để lưu lịch sử chat TRÊN GIAO DIỆN)
# Lưu ý: Đây chỉ là lịch sử hiển thị, còn logic nhớ context nằm ở Backend/Database
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Vẽ lại lịch sử chat mỗi khi reload trang
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Xử lý khi người dùng nhập liệu
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # A. Hiển thị câu hỏi của user ngay lập tức
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Gửi sang Backend xử lý
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Đang suy nghĩ...*")
        
        try:
            # Chuẩn bị dữ liệu đúng chuẩn ChatRequest của Backend
            payload = {
                "user_id": user_id,
                "query": prompt,
                "top_k": top_k
            }
            
            # GỌI API (Requests POST)
            response = requests.post(f"{BACKEND_URL}/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                
                # Hiển thị câu trả lời
                message_placeholder.markdown(answer)
                
                # Lưu vào session để chat tiếp không bị mất
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Xử lý lỗi nếu Backend trả về 404 hoặc 500
                error_msg = f"⚠️ Lỗi Backend: {response.text}"
                message_placeholder.error(error_msg)
                
        except requests.exceptions.ConnectionError:
            message_placeholder.error("🚨 Không thể kết nối tới Backend! Bạn đã chạy `uvicorn` chưa?")