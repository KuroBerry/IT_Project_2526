from src.core.Retrieval import Retrieval
from src.core.Generator import Generator
from src.core.ChatManager import ChatManager
from src.core.loader import load_components, load_user

def main():
    #Load các API, thành phần cần thiết,.....
    rewrite_model, router_model, retriever, generator = load_components()
    chat_manager = ChatManager(rewrite_model, router_model, retriever, generator)

    #Load User
    user = load_user("U001", "./users/users.json")

    import json
    print(json.dumps(user, ensure_ascii=False, indent=2))

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
            
            route, result, _ = chat_manager.handle_query(query, TOP_K)
            print(f"\n🤖 Answer ({route}): {result}")
            # print(f"\n🗂️ History ({route}): {history}")
            print("\n" + "="*50)

    except KeyboardInterrupt:
        print("\n[INFO] Dừng chương trình thủ công. Goodbye!")

if __name__ == "__main__":
    main()