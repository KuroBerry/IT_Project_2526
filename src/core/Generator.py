import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts.prompts import subjects_instructor, invalid_instructor, normal_instructor

class Generator:
    def __init__(self, model = None):
        self.model = model

    def generate_subjects(self, query, contexts):
        chain = subjects_instructor | self.model
        response = chain.invoke({"query": query, "context": contexts}).content.strip().lower()
        return response

    def generate_normal(self, query):
        """Xử lý câu hỏi hội thoại bình thường"""
        chain = normal_instructor | self.model
        response = chain.invoke({"query": query}).content.strip().lower()
        return response

    def generate_invalid(self, query):
        """Xử lý các câu hỏi invalid hoặc không rõ mục đích"""
        chain = invalid_instructor | self.model
        response = chain.invoke({"query": query}).content.strip().lower()
        return response

    def generate_guiding(self, query):
        print("Chức năng đang được phát triển...")
        return None
    
    def generate_answer(self, query, namespace, contexts):
        if namespace == "triet-hoc" or namespace == "lich-su-dang" or namespace == "tu-tuong-ho-chi-minh":
            return self.generate_subjects(query, contexts)
        elif namespace == "normal":
            return self.generate_normal(query)
        elif namespace == "invalid":
            return self.generate_invalid(query)
        elif namespace == "dinh-huong":
            return self.generate_guiding(query)