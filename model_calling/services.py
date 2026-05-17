import os
import io
import json
import subprocess
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from model_calling.schemas import PersonalityProfile, SpeechProfile

# 환경 변수 로드 및 API 클라이언트 초기화
load_dotenv()
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def process_stt(audio_bytes: bytes, filename: str = "audio.m4a") -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    
    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ko"
    )
    return response.text.strip()

# 1. 화법 추출 함수 추가 (STT 텍스트 분석)
async def extract_user_style(stt_text: str) -> dict:
    system_instruction = (
        "주어진 사용자의 대화 스크립트를 분석하여 화법 특징을 JSON 형식으로 추출하라. "
        "필수 포함 키값: frequent_words(리스트), sentence_endings(문자열), fillers(리스트), sentence_style(문자열)."
    )
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": stt_text}
        ],
        temperature=0.3
    )
    
    style_data = json.loads(response.choices[0].message.content)
    return style_data

# 2. 동적 System Prompt 생성 함수 (Big5 성격 및 추출된 화법 데이터 반영)
def build_dynamic_persona_prompt(user_persona: dict, personality: PersonalityProfile, speech: SpeechProfile) -> str:
    # schemas.py의 SpeechProfile에 user_style이 추가되었다고 가정하고 데이터 추출
    user_style = getattr(speech, 'user_style', None)
    
    if user_style:
        frequent_words = ", ".join(getattr(user_style, 'frequent_words', []))
        fillers = ", ".join(getattr(user_style, 'fillers', []))
        endings = getattr(user_style, 'sentence_endings', '')
        style_desc = getattr(user_style, 'sentence_style', '')
    else:
        frequent_words = "데이터 없음"
        fillers = "데이터 없음"
        endings = "데이터 없음"
        style_desc = "데이터 없음"

    prompt = f"""당신은 '{user_persona.get('name', '사용자')}'의 완벽한 디지털 클론입니다.
다음의 설정값과 성격, 언어 습관을 엄격하게 준수하여 대답하십시오. AI나 기계처럼 행동하지 마십시오.

[기본 정보]
- 나이: {user_persona.get('age', '알 수 없음')}
- 직업: {user_persona.get('occupation', '알 수 없음')}
- 핵심 가치관: {user_persona.get('core_values', '알 수 없음')}

[성격 파라미터 (0~100)]
- 개방성(Openness): {personality.openness}
- 성실성(Conscientiousness): {personality.conscientiousness}
- 외향성(Extraversion): {personality.extraversion}
- 친화성(Agreeableness): {personality.agreeableness}
- 신경성(Neuroticism): {personality.neuroticism}
- 성격 요약: {personality.summary}

[언어 및 발화 습관]
- 말하기 속도: {speech.speech_speed}
- 평균 음높이: {speech.avg_pitch}
- 존댓말 사용 비율: {speech.honorific_ratio}%
- 말투 요약: {speech.summary}

[사용자 고유 화법 특징 (STT 분석 기반)]
- 자주 쓰는 단어: {frequent_words}
- 종결 어미 특징: {endings}
- 주로 쓰는 추임새: {fillers} (자연스러운 위치에 배치할 것)
- 전체적인 문장 스타일: {style_desc}

사용자의 질문에 대해 위의 페르소나와 화법 특징에 완벽히 동화되어, 자연스러운 한국어로 대답하십시오."""
    return prompt

async def process_llm(user_text: str, user_persona: dict, personality: PersonalityProfile, speech: SpeechProfile) -> str:
    # 동적 프롬프트 생성 함수 호출
    system_prompt = build_dynamic_persona_prompt(user_persona, personality, speech)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    raw_text = response.choices[0].message.content.strip()
    clean_text = raw_text.replace('\n', ' ')
    
    return clean_text

# 3. Big5 성격 수치를 기반으로 ElevenLabs 파라미터 동적 계산 함수
def calculate_voice_settings(personality: PersonalityProfile):
    stability = 0.5
    style = 0.0

    # 외향성이 높을수록 억양이 다채로워짐
    if getattr(personality, 'extraversion', 50) > 60:
        style += 0.02
        stability -= 0.1
    elif getattr(personality, 'extraversion', 50) < 40:
        style += 0.0
        stability += 0.1

    # 성실성(논리성)이 높으면 stability 증가
    if getattr(personality, 'conscientiousness', 50) > 60:
        stability += 0.15
        style -= 0.005
    
    # 신경성(감정 기복)이 높으면 stability 감소
    if getattr(personality, 'neuroticism', 50) > 60:
        stability -= 0.15
        style += 0.01

    # 파라미터 유효 범위 제한
    stability = max(0.1, min(stability, 1.0))
    style = max(0.0, min(style, 1.0))

    return round(stability, 2), round(style, 2)

# process_tts 매개변수에 personality 추가
async def process_tts(
    ai_text: str,
    user_id: str,
    speech: SpeechProfile,
    personality: PersonalityProfile,
) -> str:
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    # DB(speech)에서 voice_id를 가져오되, 없으면 .env 참조
    voice_id = getattr(speech, "voice_id", None) or os.environ.get("ELEVENLABS_VOICE_ID")

    if not api_key or not voice_id:
        raise Exception("ElevenLabs API Key 또는 Voice ID가 설정되지 않았습니다.")

    # FastAPI main.py에서 /assets로 mount한 실제 폴더와 맞춘다.
    user_assets_dir = Path("model_calling") / "assets" / user_id
    user_assets_dir.mkdir(parents=True, exist_ok=True)

    # ElevenLabs API는 안정적으로 mp3를 반환받고, 최종 산출물만 m4a로 변환한다.
    temp_mp3_path = user_assets_dir / "result_audio_source.mp3"
    output_m4a_path = user_assets_dir / "result_audio.m4a"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }

    stability_val, style_val = calculate_voice_settings(personality)

    data = {
        "text": ai_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": stability_val,
            "similarity_boost": 0.9,
            "style": style_val,
            "use_speaker_boost": True,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        response = await http_client.post(url, json=data, headers=headers)

        if response.status_code != 200:
            error_detail = response.text
            raise Exception(f"ElevenLabs API 오류 [{response.status_code}]: {error_detail}")

        temp_mp3_path.write_bytes(response.content)

    # mp3 → m4a 변환
    # ffmpeg가 로컬/서버 환경에 설치되어 있어야 한다.
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_mp3_path),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(output_m4a_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise Exception(
            "ffmpeg가 설치되어 있지 않아 m4a 변환에 실패했습니다. "
            "서버 환경에 ffmpeg를 설치해 주세요."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise Exception(
            f"m4a 변환 중 ffmpeg 오류가 발생했습니다: {exc.stderr.decode(errors='ignore')}"
        ) from exc
    finally:
        if temp_mp3_path.exists():
            temp_mp3_path.unlink()

    return f"/assets/{user_id}/result_audio.m4a"


async def clone_user_voice(user_id: str, audio_bytes: bytes) -> str:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise Exception("ElevenLabs API Key가 설정되지 않았습니다.")

    url = "https://api.elevenlabs.io/v1/voices/add"
    
    headers = {
        "xi-api-key": api_key
    }
    
    files = {
        "files": ("sample.wav", audio_bytes, "audio/wav")
    }
    
    data = {
        "name": f"Clone_{user_id}",
        "description": "User customized voice clone"
    }

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(url, headers=headers, data=data, files=files)
        
        if response.status_code != 200:
            raise Exception(f"Voice cloning failed: {response.text}")
            
        response_data = response.json()
        voice_id = response_data.get("voice_id")
        
        if not voice_id:
            raise Exception("Voice ID 발급에 실패했습니다.")
            
        return voice_id