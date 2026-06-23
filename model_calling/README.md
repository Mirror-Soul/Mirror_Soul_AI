# Model Calling

## Realtime voice call

The realtime call pipeline runs automatically when `main.py` starts.

```text
WebRTC microphone track
-> utterance detection
-> Whisper STT
-> MBTI + RAG + GPT response
-> ElevenLabs TTS
-> WebRTC output audio track
```

The first implementation is turn based. A user utterance is finalized after a
short silence, then the answer is generated and played.

Required environment variables:

```env
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=

DB_HOST=
DB_PORT=3306
DB_NAME=
DB_USERNAME=
DB_PASSWORD=
```

Optional voice activity detection settings:

```env
REALTIME_VAD_ENERGY_THRESHOLD=450
REALTIME_VAD_SILENCE_SECONDS=0.8
REALTIME_VAD_MIN_SPEECH_SECONDS=0.4
REALTIME_VAD_MAX_SPEECH_SECONDS=15
```

Successful conversation logs:

```text
[WEBRTC] track received: kind=audio
[REALTIME] STT user=...: ...
[REALTIME] LLM user=...: ...
[REALTIME] reply audio queued
```

If `data/{user_id}/persona.json` exists, the realtime pipeline uses its
personality, speech style, and ElevenLabs voice ID. Otherwise it loads the
member profile and MBTI from RDS and uses `ELEVENLABS_VOICE_ID` as the default
voice.
