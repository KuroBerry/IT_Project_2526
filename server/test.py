from src.core.Retrieval import Retrieval
from src.core.Generator import Generator
from src.core.ChatManager import ChatManager
from src.core.loader import load_components
from users.user_manager import load_user
import asyncio

async def main():
    #Load User
    user = await load_user("U015")

    # import json
    # print(json.dumps(user, ensure_ascii=False, indent=2))

    #Load các API, thành phần cần thiết,.....
    multi_purposes_model, router_model, retriever, generator = load_components()

    chat_manager = ChatManager(user, multi_purposes_model, router_model, retriever, generator)
    TOP_K = 10

    # Loop vô hạn nhận query
    try:
        while True:
            query = input("\n🧠 Query: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Tạm biệt 👋")
                break
            
            result, chat_history = await chat_manager.handle_query(query, TOP_K)
            print(f"\n🤖 Answer: {result}")
            print("\n" + "="*50)

    except KeyboardInterrupt:
        print("\n[INFO] Dừng chương trình thủ công. Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())