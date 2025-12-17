import json
import os
from server.database import user_collection # Import cái collection đã kết nối ở Bước 2

# --- 1. HÀM CŨ (Giữ nguyên) ---
# Hàm này chỉ tạo dữ liệu rỗng trong RAM, không liên quan DB nên giữ nguyên
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

# --- 2. HÀM MỚI (Dùng MongoDB) ---

# Lưu ý: Có thêm từ khóa 'async' vì MongoDB (motor) chạy bất đồng bộ
async def load_user(user_id: str):
    """
    Tìm user trong Database. Nếu không có thì tạo mới và lưu luôn vào DB.
    """
    # Tìm trong DB xem có ai có _id này không
    user = await user_collection.find_one({"_id": user_id})

    if user:
        return user # Tìm thấy thì trả về ngay
    
    # Nếu chưa có: Tạo mới -> Lưu vào DB -> Trả về
    new_user = default_user(user_id)
    await user_collection.insert_one(new_user)
    print(f"✅ Đã tạo user mới trong MongoDB: {user_id}")
    
    return new_user

async def update_user_progress(new_user_info: dict):
    """
    Cập nhật thông tin user.
    Dùng $set để chỉ cập nhật những trường thay đổi (không ghi đè cả document).
    """
    user_id = new_user_info.get("_id")
    if not user_id:
        print("⚠️ Lỗi: Dữ liệu update thiếu _id")
        return

    # Lệnh update_one của Mongo
    # Tham số 1: Điều kiện tìm ({ "_id": ... })
    # Tham số 2: Dữ liệu cần sửa ({ "$set": ... })
    await user_collection.update_one(
        {"_id": user_id}, 
        {"$set": new_user_info}
    )
    print(f"💾 Đã cập nhật user {user_id} vào MongoDB")

# --- 3. HÀM TIỆN ÍCH (Giữ nguyên) ---
# Hàm get_user_level xử lý trên Dictionary trong RAM, không cần gọi DB nên giữ nguyên
def get_user_level(user, subject, level):
    subjects = user.setdefault("subjects", {})
    subject_data = subjects.setdefault(subject, {})
    level_data = subject_data.setdefault(level, {
        "progress_concepts": []
    })
    return level_data

# Hàm này đọc file nội dung môn học (Static data), không phải user data
# Nên ta tạm thời giữ nguyên logic đọc file JSON cũ.
def get_subject_content(json_path: str, subject: str, level: str):
    if not os.path.exists(json_path):
        print(f"[!] File không tồn tại: {json_path}")
        return None
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for subj in data.get("subjects", []):
        if subj.get("name") == subject:
            level_data = subj.get("level", {}).get(level)
            if level_data:
                return {
                    "subject": subj["name"],
                    "overview": subj.get("overview"),
                    "required_chapter": level_data.get("required_chapter", []),
                    "core_concepts": level_data.get("core_concepts", []),
                    "assessment_questions": level_data.get("assessment_questions", [])
                }
    return None