import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prompts.prompts import subjects_instructor, invalid_instructor, normal_instructor, guiding_instructor, extract_instructor
from users.user_manager import get_user_level, get_subject_content

class Generator:
    def __init__(self, gen_model = None):
        self.gen_model = gen_model

    #Các đoạn hội thoại liên quan đến 3 môn học đại cương
    def generate_subjects(self, query, contexts):
        chain = subjects_instructor | self.gen_model
        response = chain.invoke({"query": query, "context": contexts}).content.strip().lower()
        return response

    #Các đoạn hội thoại thông thường
    def generate_normal(self, query):
        """Xử lý câu hỏi hội thoại bình thường"""
        chain = normal_instructor | self.gen_model
        response = chain.invoke({"query": query}).content.strip().lower()
        return response

    # Xử lý các trường hợp câu trả lời invalid
    def generate_invalid(self, query):
        """Xử lý các câu hỏi invalid hoặc không rõ mục đích"""
        chain = invalid_instructor | self.gen_model
        response = chain.invoke({"query": query}).content.strip().lower()
        return response


    #Xử lý các trường hợp định hướng người học
    def generate_guiding(self, query, last_guiding, user_progress, knowledge, missing_concepts):
        chain = guiding_instructor | self.gen_model
        response = chain.invoke({"query": query, 
                                 "user_info":user_progress,
                                 "subject_requirements": knowledge, 
                                 "missing_concepts": missing_concepts, 
                                 "last_guiding": last_guiding})\
                                .content.strip().lower()
        return response

    # Hàm xử lý toàn bộ cùng 1 lúc
    def generate_answer(self, query, namespace, contexts, last_guiding, user_progress, knowledge, missing_concepts):
        if namespace == "triet-hoc" or namespace == "lich-su-dang" or namespace == "tu-tuong-ho-chi-minh":
            return self.generate_subjects(query, contexts)
        elif namespace == "normal":
            return self.generate_normal(query)
        elif namespace == "invalid":
            return self.generate_invalid(query)
        elif namespace == "dinh-huong":
            return self.generate_guiding(query, last_guiding, user_progress, knowledge, missing_concepts)