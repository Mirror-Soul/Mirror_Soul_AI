import os
import json
import traceback
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from schemas import PersonalityProfile, SpeechProfile, UserStyle
from services import process_stt, extract_user_style, process_llm, process_tts, clone_user_voice
from utils import load_user_persona

router = APIRouter()

class ChatRequest(BaseModel):
    user_text: str
    user_persona: dict
    personality: PersonalityProfile
    speech: SpeechProfile

@router.post("/extract-style", response_model=UserStyle)
async def api_extract_style(audio_file: UploadFile = File(...)):
    try:
        audio_bytes = await audio_file.read()
        stt_text = await process_stt(audio_bytes)
        style_data = await extract_user_style(stt_text)
        return UserStyle(**style_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"화법 추출 중 오류 발생: {str(e)}")

@router.post("/chat")
async def api_chat(request: ChatRequest):
    try:
        response_text = await process_llm(
            user_text=request.user_text,
            user_persona=request.user_persona,
            personality=request.personality,
            speech=request.speech
        )
        return {"response_text": response_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 생성 중 오류 발생: {str(e)}")

@router.post("/chat-voice")
async def api_chat_voice(request: ChatRequest):
    try:
        response_text = await process_llm(
            user_text=request.user_text,
            user_persona=request.user_persona,
            personality=request.personality,
            speech=request.speech
        )
        
        audio_url = await process_tts(
            ai_text=response_text,
            user_id=request.speech.user_id,
            speech=request.speech,
            personality=request.personality
        )
        return {"response_text": response_text, "audio_url": audio_url}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 대화 처리 중 오류 발생: {str(e)}")

@router.post("/register")
async def api_register(
    user_id: str = Form(...),
    user_persona_str: str = Form(..., description="JSON string format"),
    personality_str: str = Form(..., description="JSON string format"),
    speech_speed: float = Form(50.0),
    avg_pitch: float = Form(50.0),
    honorific_ratio: float = Form(50.0),
    speech_summary: str = Form("기본 말투"),
    audio_file: UploadFile = File(...)
):
    try:
        user_persona = json.loads(user_persona_str)
        personality_dict = json.loads(personality_str)
        personality = PersonalityProfile(**personality_dict)

        audio_bytes = await audio_file.read()

        voice_id = await clone_user_voice(user_id, audio_bytes)

        stt_text = await process_stt(audio_bytes)
        style_data = await extract_user_style(stt_text)
        user_style = UserStyle(**style_data)

        speech = SpeechProfile(
            user_id=user_id,
            voice_id=voice_id,
            speech_speed=speech_speed,
            avg_pitch=avg_pitch,
            honorific_ratio=honorific_ratio,
            summary=speech_summary,
            user_style=user_style
        )

        # Pydantic 버전에 따른 딕셔너리 변환 처리
        final_data = {
            "user_persona": user_persona,
            "personality": personality.model_dump() if hasattr(personality, 'model_dump') else personality.dict(),
            "speech": speech.model_dump() if hasattr(speech, 'model_dump') else speech.dict()
        }

        save_dir = f"data/{user_id}"
        os.makedirs(save_dir, exist_ok=True)
        file_path = f"{save_dir}/persona.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        return {
            "message": "회원가입 및 모델 생성이 완료되었습니다.",
            "user_id": user_id,
            "voice_id": voice_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"회원가입 처리 중 오류 발생: {str(e)}")
    
    # 타인과의 통화를 위한 요청 모델 정의
class CallRequest(BaseModel):
    target_user_id: str
    user_text: str

@router.post("/call")
async def api_call(request: CallRequest):
    # utils.py에서 파일을 찾지 못하면 내부적으로 HTTPException(404)을 발생시키므로 try 블록 밖으로 분리합니다.
    persona_data = load_user_persona(request.target_user_id)
    
    try:
        user_persona = persona_data.get("user_persona", {})
        personality = PersonalityProfile(**persona_data.get("personality", {}))
        speech = SpeechProfile(**persona_data.get("speech", {}))
        
        response_text = await process_llm(
            user_text=request.user_text,
            user_persona=user_persona,
            personality=personality,
            speech=speech
        )
        
        audio_url = await process_tts(
            ai_text=response_text,
            user_id=request.target_user_id,
            speech=speech,
            personality=personality
        )
        
        return {
            "target_user_id": request.target_user_id,
            "response_text": response_text,
            "audio_url": audio_url
        }
        
    except Exception as e:
        # 에러가 발생한 정확한 줄 번호와 원인을 터미널에 출력합니다.
        error_trace = traceback.format_exc()
        print("=== [통화 API 에러 발생 상세 로그] ===")
        print(error_trace)
        print("======================================")
        
        # str(e) 대신 repr(e)를 사용하여 숨겨져 있던 예외 클래스의 이름과 내용을 모두 반환합니다.
        raise HTTPException(status_code=500, detail=f"통화 처리 중 오류 발생: {repr(e)}")