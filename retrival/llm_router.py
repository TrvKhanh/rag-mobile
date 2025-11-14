import ast
from generation.llm_stm import LLM
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

model = LLM | StrOutputParser()

system_prompt = """ Bạn là bộ phân loại yêu cầu của người dùng cho hệ thống chatbot RAG điện thoại.
                        Nhiệm vụ:
                        1. Phân tích câu hỏi hoặc lời nói của người dùng.
                        2. Chuẩn hóa và tóm tắt nội dung chính thành chuỗi ngắn gọn (trường "infor") nếu thông tin liên quan đến sản phẩm thì 
                        chuẩn hóa sao cho chỉ liên quan đến truy vấn ví dụ : tôi muốn mua iphone 16, chuẩn hóa thành thông tin chi tiết iphone 16,loại bỏ những từ gây
                         nhiểu khiến truy vấn kém chính xác, chuyển câu về kí tự thường , không viết hoa .
                        3. Luôn xuất kết quả ở định dạng Dict hợp lệ, không thêm giải thích.

                        **Quy tắc xác định router**:
                        - Nếu chỉ là cảm xúc, lời chào, hỏi thăm → "chatchit".
                        - Nếu có yêu cầu tìm hiểu, hỏi thông tin, tra cứu, hoặc liên quan tới sản phẩm (giá cả, thời điểm ra mắt , thông tin màn hình ...) → "rag".
                        - Quan trọng : chỉ trả về dict python không thêm giải thích

                        Đầu ra Dict mẫu:

                        {"infor": "<nội dung chính>", "router": "<chatchit hoặc rag>"}

                        Ví dụ:
                        User: "Chào bạn hôm nay là ngày đẹp trời nhỉ"
                        Output: {"infor": "ngày đẹp trời", "router": "chatchit"}

                        User: "Chào bạn hôm nay đẹp trời quá nên muốn mua một con iPhone 16"
                        Output: {"infor": "thông tin chi tiết iPhone 16", "router": "rag"}

                        Bây giờ, hãy xử lý câu nhập tiếp theo:
                        """     
    
def llm_router(user_input: str, system_prompt=system_prompt) -> dict:
    list_router = []
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    content = model.invoke(messages)
    try:
        router_dict = ast.literal_eval(content)
    
    except Exception:
        router_dict = {"infor": None, "router": None}
    return router_dict