import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/edurag_db")
DB_NAME = "edurag_db"

print(f"🔗 Kết nối đến MongoDB tại: {MONGO_URI}")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

user_collection = db["users"]