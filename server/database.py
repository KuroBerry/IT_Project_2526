import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME")
USER_COLLECTION = os.environ.get("USER_COLLECTION")

# 2. Tạo Client kết nối
client = AsyncIOMotorClient(MONGO_URL)

# 3. Chọn Database
db = client[DB_NAME]

# 4. Chọn Collection (ngăn chứa Users)
user_collection = db.get_collection(USER_COLLECTION)

print(f"\t[INFO]: Connected to MongoDB at: {MONGO_URL} | DB: {DB_NAME}")