import json
from langchain_core.tools import tool
from typing import List, Dict

# Path to the store data
STORES_FILE_PATH = "data/stores.json"

@tool
def find_nearby_stores(city: str) -> str:
    """
    Use this tool to find the addresses of store branches in a specific city.
    The input must be a city name in Vietnam, for example: "Hà Nội" or "Hồ Chí Minh".
    """
    try:
        with open(STORES_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return "Lỗi: Không tìm thấy file dữ liệu cửa hàng."
    except json.JSONDecodeError:
        return "Lỗi: File dữ liệu cửa hàng bị hỏng."

    city_normalized = city.strip().lower()
    
    found_stores: List[Dict] = [
        store for store in data.get("stores", []) 
        if city_normalized in store.get("city", "").lower()
    ]
    
    if not found_stores:
        return f"Rất tiếc, không tìm thấy cửa hàng nào ở '{city}'."
        
    response = f"Tìm thấy {len(found_stores)} cửa hàng ở {city}:\n"
    for store in found_stores:
        response += f"- {store['address']}\n"
        
    return response
