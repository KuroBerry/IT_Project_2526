from prompts.prompts import router_prompt
from utils import summarize_answer


class ChatManager:
    def __init__(self, router_model, retriever, generator):
        self.router_model = router_model
        self.retriever = retriever
        self.generator = generator
        self.history = []   # lưu lịch sử hội thoại hiện tại

    # ========================================
    # 1️⃣ Phân loại truy vấn (route detection)
    # ========================================
    def query_router(self, query):
        chain = router_prompt | self.router_model
        return chain.invoke({"query": query}).content.strip().lower()

    # ========================================
    # 2️⃣ Quản lý lịch sử chat
    # ========================================
    def add_to_history(self, user_query, bot_answer):
        # Tóm tắt câu trả lời trước khi lưu
        sum_ans = summarize_answer(bot_answer, self.router_model)
        # print(f"[INFO] Tóm tắt câu trả lời để lưu lịch sử: {sum_ans}")
        self.history.append({"user": user_query, "bot": sum_ans})

    def get_recent_history(self, limit=30):
        """Lấy n lượt hội thoại gần nhất"""
        return self.history[-limit:]

    # ========================================
    # 3️⃣ Ghép prompt có lịch sử
    # ========================================
    def build_context_prompt(self, user_query):
        if not self.history:
            return user_query
        history_text = "\n".join(
            [f"Người dùng: {h['user']}\nTrợ lý: {h['bot']}" for h in self.get_recent_history()]
        )
        return f"""Dưới đây là lịch sử trò chuyện gần đây:
                {history_text}

                Câu hỏi mới: {user_query}
                Hãy trả lời dựa trên ngữ cảnh trên.
                """

    # ========================================
    # 4️⃣ Xử lý truy vấn chính (RAG + Memory)
    # ========================================
    def handle_query(self, query):
        # B1: Phân loại
        route = self.query_router(query)

        # B2: Tìm ngữ cảnh nếu thuộc domain RAG
        contexts = None
        if route in ["lich-su-dang", "triet-hoc", "tu-tuong-ho-chi-minh"]:
            contexts = self.retriever.hybrid_search(query, route, 10)

        # B3: Thêm lịch sử vào prompt
        enriched_query = self.build_context_prompt(query)
        
        # B4: Sinh câu trả lời
        result = self.generator.generate_answer(enriched_query, route, contexts)

        # B5: Lưu lại vào lịch sử
        self.add_to_history(query, result)

        return route, result, self.history

    # ========================================
    # 5️⃣ Tiện ích
    # ========================================
    def clear_history(self):
        self.history = []
        print("[INFO] Đã xóa lịch sử hội thoại.")