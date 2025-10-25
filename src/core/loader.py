#Đoạn này là để load các thành phần cần thiết như là mô hình embedding, kết nối Pinecone, mô hình Gemini, và khởi tạo các thành phần Retrieval và Generation.
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from core.Retrieval import Retrieval
from core.Generator import Generator

from utils import get_bm25_vocabulary
from config.setting import settings


def load_components():
    print("[INFO] Loading embedding model and BM25 vocabulary...")
    embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
    bm25, vocabulary = get_bm25_vocabulary()

    print("[INFO] Connecting to Pinecone...")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    dense_index = pc.Index(host=settings.HOST_DENSE)
    sparse_index = pc.Index(host=settings.HOST_SPARSE)

    print("[INFO] Connecting to Gemini Model...")
    router_model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai") 
    generator_model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

    print("[INFO] Initializing Retrieval and Generation components...")
    retriever = Retrieval(pc, dense_index, sparse_index, embedding_model, bm25, vocabulary)
    generator = Generator(generator_model)

    return router_model, retriever, generator
