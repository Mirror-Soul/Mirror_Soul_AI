import os
import json
from fastapi import HTTPException

def load_user_persona(user_id: str) -> dict:
    file_path = f"data/{user_id}/persona.json"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"사용자 {user_id}의 페르소나 설정을 찾을 수 없습니다.")
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 로드 중 오류 발생: {str(e)}")