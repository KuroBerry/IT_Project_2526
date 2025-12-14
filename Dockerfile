# 1. Dùng Python 3.9 bản rút gọn
FROM python:3.12.7-slim

# 2. Tạo thư mục làm việc
WORKDIR /app

# 3. Copy file thư viện vào trước (để tận dụng cache)
COPY requirements.txt .

# 4. Cài đặt thư viện (ép không dùng cache của pip để giảm dung lượng image)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code hiện tại vào
COPY . .

# 6. Mặc định giữ container sống để chờ lệnh (quan trọng cho việc test)
CMD ["tail", "-f", "/dev/null"]