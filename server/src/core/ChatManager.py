from prompts.prompts import router_prompt, rewrite_prompt, extract_concepts_instructor, extract_instructor
from utils import summarize_answer
from server.users.user_manager import get_user_level, get_subject_content, update_user_progress
import json


class ChatManager:
    def __init__(self, user, multi_purposes_model, router_model, retriever, generator):
        self.multi_purposes_model = multi_purposes_model
        self.router_model = router_model
        self.retriever = retriever
        self.generator = generator
        self.user = user
        self.history = []   # lưu lịch sử hội thoại hiện tại

    # ========================================
    # 1 Viết lại truy vấn đầy đủ ngữ cảnh
    # ========================================
    def rewrite_query(self, query, history):
        """
        Hàm sẽ sử dụng LLMs để viết lại truy vấn của người dùng một cách đầy đủ ngữ cảnh hơn.
        vd: "Ông ra đi tìm đường cứu nước vào năm nào, ở đâu?"  ---(tùy vào lịch sử)---> "Chủ tịch Hồ Chí Minh ra đi tìm đường cứu nước vào năm nào, ở đâu?"
        """
        chain = rewrite_prompt | self.multi_purposes_model
        return chain.invoke({"query": query, "history": history}).content.strip().lower()

    # ========================================
    # 2 Phân loại truy vấn (route detection)
    # ========================================
    def query_router(self, query):
        chain = router_prompt | self.router_model
        return chain.invoke({"query": query}).content.strip().lower()
    
    # ========================================
    # 3 Xác định môn học và cấp độ người dùng cần định hướng
    # ========================================
    def extract_subject_and_level(self, query):
        chain = extract_instructor | self.multi_purposes_model
        result = chain.invoke({"query": query, "last_guiding": self.user["last_guiding"]}).content.strip()

        # Tách kết quả theo dấu phẩy
        parts = [p.strip() for p in result.split(",")]

        # Đảm bảo có đủ 2 phần (subject, level)
        subject = parts[0] if len(parts) > 0 else None
        level = parts[1] if len(parts) > 1 else None

        return subject, level
    
    # ========================================
    # 4 Tìm kiếm các khái niệm của môn học mà người dùng chưa học xong trong lộ trình hiện tại của họ
    # ========================================
    def find_missing_concepts(self):
        subject = self.user["last_guiding"].get("subject")
        level = self.user["last_guiding"].get("level")

        if subject != "None" and level != "None":
            user_progress = get_user_level(self.user, subject, level)
            knowledge = get_subject_content("./server/data/Knowledge/guiding.json", subject, level)

            #Tính toán các khái niệm mà người dùng còn chưa học xong
            missing_concepts = set(knowledge["core_concepts"]) - set(user_progress["progress_concepts"])
            return user_progress, knowledge, missing_concepts
        else:
            return None, None, None
    
    # ========================================
    # 5 Trích xuất các khái niệm mà người dùng đã học sau 1 phiên tương tác
    # ========================================
    def extract_concept(self, user_query, bot_answer, missing_concepts):

        # Lấy lịch sử hội thoại gần đây (ví dụ 5 lượt)
        chat_history = self.get_recent_history(5)
        
        # Ghép prompt cho chain
        chain = extract_concepts_instructor | self.multi_purposes_model

        # Gọi mô hình
        response = chain.invoke({
            "chat_history": chat_history,
            "user_query": user_query,
            "bot_answer": bot_answer,
            "missing_concepts": missing_concepts
        }).content.strip()

        # Chuẩn hóa kết quả: nếu không có gì hoặc trả về "none" → chuyển về None thực
        if not response or response.lower() == "none":
            return None

        return response

    # ========================================
    # 6 Quản lý thông tin người dùng sau mỗi phiên chat
    # ========================================
    def post_interaction(self, user_query, bot_answer):
        #Cập nhật thông tin của người dùng sau mỗi phiên chat (Mỗi lần tương tác)

        #Tóm tắt câu trả lời gần đây nhất
        sum_ans = summarize_answer(bot_answer, self.multi_purposes_model)

        #Tạo record của đoạn hội thoại mới nhất
        new_conversation = [
            {"role": "user", "content": user_query},
            {"role": "bot", "content": sum_ans}
        ]

        #Cập nhật lịch sử chat của người dùng
        self.user.setdefault("chat_history", [])
        self.user["chat_history"].extend(new_conversation)
        #Chỉ lấy 20 đoạn hội thoại gần đây nhất (Do mỗi đoạn gồn 2 entry)
        self.user["chat_history"] = self.user["chat_history"][-40:]


        subject = self.user.get("last_guiding", {}).get("subject")
        level = self.user.get("last_guiding", {}).get("level")

        #Cập nhật các khái niệm mà người dùng đã hoàn thành trong các phiên chat gần đây
        _, _, missing_concepts = self.find_missing_concepts()
                
        new_concepts = self.extract_concept(user_query,
                                            bot_answer,
                                            missing_concepts)

        #Thêm các khái niệm người dùng vừa học được:
        if new_concepts is not None and new_concepts in missing_concepts and missing_concepts is not None:
            progress_path = self.user.setdefault("subjects", {}) \
                        .setdefault(subject, {}) \
                        .setdefault(level, {}) \
                        .setdefault("progress_concepts", [])
            progress_path.append(new_concepts)
        

    def get_recent_history(self, limit=20):
        """Lấy n lượt hội thoại gần nhất."""
        history = self.user.get("chat_history", [])
        return history[-limit:] if len(history) > 0 else []
    
    # ========================================
    # 7 Ghép prompt có lịch sử
    # ========================================
    def build_context_prompt(self, user_query):
        """Xây prompt ngữ cảnh từ lịch sử hội thoại gần nhất."""
        history = self.get_recent_history()

        # Ghép thành các cặp: user -> bot
        history_lines = []
        i = 0
        while i < len(history):
            turn = history[i]
            if turn["role"] == "user":
                user_text = turn["content"]
                bot_text = history[i + 1]["content"] if i + 1 < len(history) and history[i + 1]["role"] == "bot" else ""
                history_lines.append(f"Người dùng: {user_text}\nTrợ lý: {bot_text}")
                i += 2
            else:
                i += 1  # bỏ qua nếu có lệch vai trò

        history_text = "\n\n".join(history_lines) if history_lines else "(chưa có hội thoại trước đó)"

        prompt = f"""Dưới đây là lịch sử trò chuyện gần đây:
                    {history_text}

                    Câu hỏi mới: {user_query}
                    Hãy trả lời dựa trên ngữ cảnh trên.
                    """
        return prompt

    # ========================================
    # 8 Xử lý truy vấn chính (RAG + Memory)
    # ========================================
    async def handle_query(self, query, top_k):

        # B1: Viết lại Query đủ ngữ cảnh và Phân loại
        recent_conversation = self.get_recent_history(5) #Lấy ra lịch sử 5 cuộc hội thoại gần nhất
        rewrite_query = self.rewrite_query(query, recent_conversation)
        print("\t[INFO] Đang viết lại câu đầy đủ ngữ cảnh")

        # Phân loại truy vấn để xác định route
        route = self.query_router(rewrite_query)
        print("\t[INFO] Phát hiện chức năng:", route)

        # Kết hợp promt hiện tại cùng với lịch sử chat
        enriched_query = self.build_context_prompt(rewrite_query)
        print("\t[INFO] Kết hợp câu vào với lịch sử chat trước đó")

        # Xử lý các route tương ứng
        contexts = None
        result = None
        user_progress = None
        knowledge = None
        missing_concepts = None

        if route in ["lich-su-dang", "triet-hoc", "tu-tuong-ho-chi-minh"]:
            print("\t[INFO] Đang thực hiện tìm kiếm nội dung theo câu hỏi")
            contexts = self.retriever.hybrid_search(rewrite_query,
                                                    route,
                                                    top_k)
        elif route == "dinh-huong":
            print("\t[INFO] Đang lấy thông tin từ lộ trình học của người dùng")
            subject, level = self.extract_subject_and_level(enriched_query)

            #Nễu không xác định được level hay môn học cần định hướng
            if subject == "None" or level == "None":
                result = "Còn thiếu thông tin cần thiết"
                self.post_interaction(query, result)
                return result, self.history
            
            # Cập nhật last_guiding của người dùng
            self.user["last_guiding"] = {"subject": subject, "level": level}

            # Lấy ra lộ trình hiện tại, các khái niệm cần thiết của môn học và các khái niệm người dùng chưa học xong
            user_progress, knowledge, missing_concepts = self.find_missing_concepts()

        # B4: Sinh câu trả lời
        print(f"\t[INFO] Đang suy nghĩ câu trả lời...")
        # print(user["last_guiding"])
        result = self.generator.generate_answer(query = enriched_query, 
                                                namespace = route, 
                                                contexts = contexts, 
                                                last_guiding = self.user["last_guiding"],
                                                user_progress = user_progress, 
                                                knowledge = knowledge, 
                                                missing_concepts = missing_concepts)

        # B5: Xử lý sau mỗi lần tương tác
        self.post_interaction(query, result)
        await update_user_progress(self.user)
        print("="*50)

        return result, self.history