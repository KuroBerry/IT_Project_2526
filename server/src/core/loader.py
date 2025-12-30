#Đoạn này là để load các thành phần cần thiết như là mô hình embedding, kết nối Pinecone, mô hình Gemini, và khởi tạo các thành phần Retrieval và Generation.
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from core.Retrieval import Retrieval
from core.Generator import Generator

from utils import get_bm25_vocabulary, save_chunks_to_json
from config.setting import settings

from os import getenv
from dotenv import load_dotenv

load_dotenv()

#Hàm dùng để load các thành phần cần thiết để triên khai dự án
def load_components():
    print("[INFO] Loading embedding model and BM25 vocabulary...")
    embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
    bm25, vocabulary = get_bm25_vocabulary()

    print("[INFO] Connecting to Pinecone...")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    dense_index = pc.Index(host=settings.HOST_DENSE)
    sparse_index = pc.Index(host=settings.HOST_SPARSE)

    print("[INFO] Connecting to Gemini 2.5 Flash Lite Model...")
    multi_purposes_model = init_chat_model("google/gemini-2.5-flash-lite", model_provider="openai",api_key=getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    router_model = init_chat_model("google/gemini-2.5-flash-lite", model_provider="openai", api_key=getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    generator_model = init_chat_model("google/gemini-2.5-flash-lite", model_provider="openai",api_key=getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

    print("[INFO] Initializing Retrieval and Generation components...")
    retriever = Retrieval(pc, dense_index, sparse_index, embedding_model, bm25, vocabulary)
    generator = Generator(generator_model)

    print("[INFO] Loading knowlege info...")
    

    return multi_purposes_model, router_model, retriever, generator
