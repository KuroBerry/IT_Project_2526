# import json
# import os

# FILE_PATH = "./server/users/user_demo.json"

# def save_chunks_to_json(chunks, output_path):
#     """
#     Lưu danh sách các chunk (list of dict) ra file JSON.
#     """
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Tạo folder nếu chưa có

#     with open(output_path, "w", encoding="utf-8", errors='ignore') as f:
#         json.dump(chunks, f, ensure_ascii=False, indent=2)

#     print(f"✅ Đã lưu {len(chunks)} chunks vào: {output_path}")

# #Tạo user trống với 3 môn, mỗi môn sẽ có 3 cấp độ trong đó
# def default_user(user_id: str):
#     levels = ["beginner", "exam", "advanced"]
#     subjects = ["lich-su-dang", "tu-tuong-ho-chi-minh", "triet-hoc"]

#     return {
#         "_id": user_id,
#         "name": "",
#         "last_guiding": {
#             "subject": "None",
#             "level": "None"
#         },
#         "chat_history": [],
#         "subjects": {
#             subject: {
#                 level: {
#                     "progress_concepts": []
#                 }
#                 for level in levels
#             }
#             for subject in subjects
#         }
#     }

# #Load người dùng từ dữ liệu, nếu không có thì tạo mới và lưu lại
# def load_user(user_id: str):
#     file_path = FILE_PATH
#     users = []

#     # --- Bước 1: Đọc dữ liệu file ---
#     if os.path.exists(file_path):
#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 data = f.read().strip()
#                 if data:  # tránh lỗi file rỗng
#                     users = json.loads(data)
#         except (json.JSONDecodeError, FileNotFoundError):
#             users = []  # nếu lỗi đọc JSON, reset danh sách

#     # --- Bước 2: Tìm user theo ID ---
#     for user in users:
#         if user.get("_id") == user_id:
#             # Đảm bảo có đủ key (phòng lỗi khi format cũ)
#             user.setdefault("last_guiding", {})
#             user.setdefault("chat_history", [])
#             user.setdefault("subjects", default_user(user_id)["subjects"])
#             return user

#     # --- Bước 3: Nếu không có thì tạo mới ---
#     new_user = default_user(user_id)
#     users.append(new_user)

#     # --- Bước 4: Lưu lại file ---
#     save_chunks_to_json(users, file_path)

#     return new_user

# # Lấy thông tin môn học và cấp độ của người dùng
# def get_user_level(user, subject, level):
#     subjects = user.setdefault("subjects", {})
#     subject_data = subjects.setdefault(subject, {})
#     level_data = subject_data.setdefault(level, {
#         "progress_concepts": []
#     })
#     return level_data

# # Lấy thông tin môn học và cấp độ từ kho tri thức
# def get_subject_content(json_path: str, subject: str, level: str):
#     # Đọc dữ liệu JSON
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Tìm môn học
#     for subj in data.get("subjects", []):
#         if subj.get("name") == subject:
#             # Kiểm tra cấp độ
#             level_data = subj.get("level", {}).get(level)
#             if level_data:
#                 return {
#                     "subject": subj["name"],
#                     "overview": subj.get("overview"),
#                     "required_chapter": level_data.get("required_chapter", []),
#                     "core_concepts": level_data.get("core_concepts", []),
#                     "assessment_questions": level_data.get("assessment_questions", [])
#                 }
#             else:
#                 print(f"[!] Không tìm thấy cấp độ '{level}' trong môn '{subject}'.")
#                 return None

#     print(f"[!] Không tìm thấy môn học '{subject}'.")
#     return None

# # Cập nhật tiến độ học tập của người dùng 
# def update_user_progress(new_user_info):

#     #Đọc qua dữ liệu file
#     if os.path.exists(FILE_PATH):
#         try:
#             with open(FILE_PATH, "r", encoding="utf-8") as f:
#                 data = f.read().strip()
#                 if data:  # tránh lỗi file rỗng
#                     users = json.loads(data)
#         except (json.JSONDecodeError, FileNotFoundError):
#             users = []  # nếu lỗi đọc JSON, reset danh sách

#     #Cập nhật dữ liệu user
#     for user in users:
#         if user.get("_id") == new_user_info.get("_id"):
#             user.update(new_user_info)
#             # print(user)
    
#     #Lưu lại file
#     save_chunks_to_json(users, FILE_PATH)




















import json
import os
from server.database import user_collection

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
    print(new_user_info)
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