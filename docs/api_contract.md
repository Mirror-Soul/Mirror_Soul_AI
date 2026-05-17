# AI API Contract

## Training API

### POST /api/v1/training/samples

백엔드가 AI 서버로 회원 학습 데이터를 전달합니다.

Request:
```json
{
  "userId": "user_123",
  "aiProfileId": "profile_user_123",
  "mbti": "ENFJ",
  "questionId": 1,
  "questionCategory": "외향성",
  "questionText": "...",
  "transcript": "...",
  "audioUrl": "..."
}