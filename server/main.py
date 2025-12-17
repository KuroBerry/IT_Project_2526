from database import user_collection # Import cái collection đã kết nối ở Bước 2
import asyncio

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

async def load_user(user_id: str):
    """
    Tìm user trong Database. Nếu không có thì tạo mới và lưu luôn vào DB.
    """
    # Tìm trong DB xem có ai có _id này không
    user = await user_collection.find_one({"_id": user_id})

    if user:
        print(user['name'])
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

if __name__ == "__main__":
    asyncio.run(load_user("U002"))