import json

def get_user_level(user, subject, level):
    """
    Lấy thông tin môn học theo tên và cấp độ (level).

    Args:
        user (dict): thông tin người dùng, có key 'subjects'
        subject (str): tên môn học
        level (str): level mong muốn ('beginner', 'exam', 'advanced')

    Returns:
        dict: dữ liệu của môn học ở level tương ứng
    """
    subjects = user.setdefault("subjects", {})
    subject_data = subjects.setdefault(subject, {})
    level_data = subject_data.setdefault(level, {
        "progress_concepts": []
    })
    return level_data


import json

def get_subject_content(json_path: str, subject: str, level: str):
    """
    Lấy thông tin học tập theo môn và cấp độ.

    Args:
        json_path (str): đường dẫn tới file JSON dữ liệu.
        subject (str): tên hoặc mã môn học (vd: 'triet-hoc').
        level (str): cấp độ ('beginner', 'exam', 'advanced').

    Returns:
        dict: chứa overview, required_chapter, core_concepts, assessment_questions.
              Nếu không tìm thấy thì trả về None.
    """
    # Đọc dữ liệu JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tìm môn học
    for subj in data.get("subjects", []):
        if subj.get("name") == subject:
            # Kiểm tra cấp độ
            level_data = subj.get("level", {}).get(level)
            if level_data:
                return {
                    "subject": subj["name"],
                    "overview": subj.get("overview"),
                    "required_chapter": level_data.get("required_chapter", []),
                    "core_concepts": level_data.get("core_concepts", []),
                    "assessment_questions": level_data.get("assessment_questions", [])
                }
            else:
                print(f"[!] Không tìm thấy cấp độ '{level}' trong môn '{subject}'.")
                return None

    print(f"[!] Không tìm thấy môn học '{subject}'.")
    return None
