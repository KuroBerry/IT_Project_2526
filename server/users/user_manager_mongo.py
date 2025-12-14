import os
from server.database import users_collection 

# --- HÀM 1: Tạo cấu trúc User mặc định (GIỮ NGUYÊN) ---
# Logic này vẫn cần thiết để khởi tạo dữ liệu cho người mới
def default_user(user_id: str):
    levels = ["beginner", "exam", "advanced"]
    subjects = ["lich-su-dang", "tu-tuong-ho-chi-minh", "triet-hoc"]

    return {
        "_id": user_id, # MongoDB dùng _id làm khóa chính luôn
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

# --- HÀM 2: Load User (THAY ĐỔI LỚN) ---
def load_user(user_id: str):
    """
    Tìm user trong MongoDB. Nếu không có thì tạo mới và lưu vào DB.
    """
    # 1. Tìm trong DB
    user = users_collection.find_one({"_id": user_id})

    # 2. Nếu tìm thấy -> Trả về luôn
    if user:
        # (Optional) Validate cấu trúc dữ liệu cũ nếu cần thiết tại đây
        return user

    # 3. Nếu chưa có -> Tạo mới và Insert vào DB
    print(f"👤 Tạo user mới: {user_id}")
    new_user = default_user(user_id)
    users_collection.insert_one(new_user)
    
    return new_user

# --- HÀM 3: Update Progress (THAY ĐỔI LỚN) ---
def update_user_progress(new_user_info):
    """
    Cập nhật toàn bộ thông tin user vào DB
    """
    user_id = new_user_info.get("_id")
    if not user_id:
        print("❌ Lỗi: User data không có _id")
        return

    # Dùng lệnh replace_one để ghi đè thông tin mới vào user cũ
    # upsert=True: Nếu lỡ user bị xóa mất thì tự tạo lại
    users_collection.replace_one({"_id": user_id}, new_user_info, upsert=True)
    
    # Debug log (tắt đi khi chạy thật)
    # print(f"✅ Đã cập nhật tiến độ cho user: {user_id}")

# --- HÀM 4: Helper lấy Level (GIỮ NGUYÊN) ---
# Vì 'user' load từ Mongo về vẫn là Dictionary Python, nên hàm này không đổi
def get_user_level(user, subject, level):
    subjects = user.setdefault("subjects", {})
    subject_data = subjects.setdefault(subject, {})
    level_data = subject_data.setdefault(level, {
        "progress_concepts": []
    })
    return level_data