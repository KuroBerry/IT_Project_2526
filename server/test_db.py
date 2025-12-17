import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# --- 1. CẤU HÌNH KẾT NỐI ---
# Vì bạn chạy file này từ máy thật (host) kết nối vào Docker, nên dùng 'localhost'
# Cấu trúc: mongodb://user:pass@host:port
MONGO_URL = "mongodb://admin:123456@localhost:27017"
DB_NAME = "edurag_db"

async def main():
    print("⏳ Đang bắt đầu kết nối...")

    # --- 2. TẠO CLIENT (Người vận chuyển) ---
    # Client này chịu trách nhiệm duy trì kết nối với MongoDB
    client = AsyncIOMotorClient(MONGO_URL)

    try:
        # --- 3. CHỌN DATABASE (Ngăn tủ lớn) ---
        db = client[DB_NAME]

        # --- 4. CHỌN COLLECTION (Ngăn kéo chứa Users) ---
        collection = db["user_demo"]
        subject_collection = db["subject_knowledge"]

        # --- 5. TRUY VẤN (Đọc dữ liệu) ---
        # Lệnh find_one(): Lấy về 1 người bất kỳ đầu tiên nó thấy
        print("🔍 Đang tìm kiếm user trong collection 'users'...")
        user = await collection.find_one({"_id": "U001"})

        # --- 6. HIỂN THỊ KẾT QUẢ ---
        if user:
            print("\n" + "="*40)
            print("✅ KẾT NỐI THÀNH CÔNG! Đã lấy được dữ liệu:")
            print("="*40)
            print(f"👤 User ID: {user.get('_id')}")
            print(f"👤 Name   : {user.get('name')}")
            print("-" * 20)
        else:
            print("\n⚠️ Kết nối được nhưng Collection đang RỖNG (chưa có dữ liệu).")

    except Exception as e:
        print(f"\n❌ LỖI KẾT NỐI: {e}")

# Chạy chương trình bất đồng bộ
if __name__ == "__main__":
    asyncio.run(main())