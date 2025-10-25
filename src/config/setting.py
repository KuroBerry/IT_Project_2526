import os
from dotenv import load_dotenv

load_dotenv("../.env")

class Settings:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    HOST_DENSE = os.getenv("HOST_DENSE")
    HOST_SPARSE = os.getenv("HOST_SPARSE")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

settings = Settings()