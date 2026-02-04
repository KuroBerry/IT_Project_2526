# EduRAG - Hệ thống Trợ giảng Ảo (RAG Chatbot)

Dự án Chatbot hỗ trợ học tập các môn đại cương (Triết học, Lịch sử Đảng...), sử dụng kiến trúc RAG (Retrieval-Augmented Generation) kết hợp với Google Gemini và Pinecone Vector DB.

## 🛠 Công nghệ sử dụng
* **Backend:** FastAPI (Python)
* **Frontend:** Streamlit
* **Database:** MongoDB
* **DevOps:** Docker & Docker Compose

---

## ⚙️ Yêu cầu Cài đặt (Prerequisites)

Để chạy được dự án, máy tính của bạn cần cài đặt các phần mềm sau:

1.  **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Bắt buộc): Để chạy môi trường server ảo hóa.
2.  **[Git](https://git-scm.com/downloads)**: Để quản lý mã nguồn.
3.  **[MongoDB Compass](https://www.mongodb.com/try/download/compass)** (Quan trọng): Công cụ giao diện để xem và quản lý dữ liệu trong Database.


---

## 🚀 Hướng dẫn Chạy dự án (Quick Start)

### Bước 1: Clone dự án về máy
```bash
git clone git@github.com:KuroBerry/IT_Project_2526.git
cd IT_Project_2526
```
### Bước 2: Cấu hình biến môi trường (.env)
Vì lý do bảo mật, file chứa mật khẩu sẽ không có trên Git. Bạn cần tự tạo nó.

Tìm file .env.example trong thư mục gốc.

Copy nó và đổi tên thành .env.

Mở file .env lên và điền các Key của bạn vào (giữ nguyên cấu hình Mongo ở dưới để chạy được cả Local và Docker).

Nội dung chuẩn của file .env:

```bash
# --- DATABASE CONFIG (User/Pass mặc định) ---
MONGO_INIT_USER=admin
MONGO_INIT_PASS=123456
DB_NAME=edurag_db

# --- CONNECTION STRING (Quan trọng) ---
# Dùng localhost để bạn có thể kết nối từ Compass hoặc chạy script test tay
MONGO_URL=mongodb://admin:123456@localhost:27017/edurag_db?authSource=admin

# --- API KEYS (Điền key thật của bạn vào đây) ---
PINECONE_API_KEY=your_pinecone_key_here
HOST_DENSE=your_host_dense_here
HOST_SPARSE=your_host_sparse_here
GOOGLE_API_KEY=your_google_key_here
API_URL=http://backend:8000
```

### Bước 3: Khởi chạy hệ thống
Mở Terminal tại thư mục dự án và chạy lệnh

```bash
docker-compose up -d --build
```
(Lệnh này sẽ tự động tải thư viện, dựng Database và bật Server. Quá trình đầu tiên có thể mất vài phút).

## 🗄 Hướng dẫn Nạp dữ liệu (Database Setup)
Khi chạy lần đầu, Database trong Docker sẽ trống trơn. Bạn cần nạp dữ liệu mẫu vào thì App mới chạy được.

### 1. Kết nối MongoDB Compass
- Mở phần mềm MongoDB Compass.

- Dán chuỗi kết nối sau vào ô URI:
```bash
mongodb://admin:123456@localhost:27017/?authSource=admin
```
- Bấm **Connect**
### 2. Import dữ liệu mẫu
- Sau khi kết nối, bấm nút dấu (+) cạnh chữ Database để tạo DB mới.
    - Database Name: `edurag_db`
    - Collection Name: `user_demo`
- Vào collection user vừa tạo.

- Bấm nút Import Data (màu xanh lá).
- Chọn file: server/users/users.json trong thư mục code.
- Chọn format JSON và bấm Import.

👉 Xong! Giờ bạn có thể vào Chatbot để test.

## 🌐 Truy cập ứng dụng
- Frontend (Chatbot): http://localhost:8501
- Backend API Docs: http://localhost:8000/docs
