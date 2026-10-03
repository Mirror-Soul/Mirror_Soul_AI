# Call Context API Rollout

## Automated verification (2026-10-03)

- Result: PASS
- Command: `OPENAI_API_KEY=test-key python -m unittest discover -s tests -t .`
- Tests: 150 passed
- Runtime DB search: no direct clone/profile/voice lookup remains in signaling,
  WebRTC, or realtime turn processing. The voice training worker intentionally
  retains its database access.

## Live smoke test

- Result: NOT RUN
- Reason: the local workspace has no call-server `.env`,
  `BACKEND_API_BASE_URL`, or `AI_INTERNAL_API_KEY`, and no READY test call was
  supplied.

Run this only with a real READY call and production-equivalent secrets:

```bash
curl \
  --fail-with-body \
  -H "X-Internal-Api-Key: $AI_INTERNAL_API_KEY" \
  "$BACKEND_API_BASE_URL/internal/ai/calls/$CALL_ID/context"
```

Then verify one VOICE call and one VIDEO call:

- `CALL_INVITE.data` and `CALL_ACCEPT.data` contain only `callId`.
- The backend context endpoint is called once per call.
- VOICE uses the response `voiceId` and creates no video output track.
- VIDEO uses the response `mediaType` and creates the video output track.
- No MySQL connection appears in call-server logs.
- `CALL_END` or peer close removes the cached context.
- Neither the API key nor the complete voice ID appears in logs.

## Deployment window

1. Configure the same `AI_INTERNAL_API_KEY` on the backend and call server.
2. Configure `BACKEND_API_BASE_URL` and
   `BACKEND_CALL_CONTEXT_TIMEOUT_SECONDS=5` on the call server.
3. Confirm the backend context endpoint with a READY call.
4. Deploy backend signaling, AI call server, and frontend contract changes in
   the same window.
5. Restart `mirror-soul-call` and confirm it is active.
6. Complete and record the VOICE and VIDEO smoke checks above.

## Removing database credentials from AI servers

After `codex/backend-db-read-decoupling`, no AI component opens a MySQL
connection: the call server uses the call context API, the training API writes
only to ChromaDB, the voice and face workers report through SQS, and the face
member-voice preview uses a fixed fallback voice instead of reading
`ai_voice_profiles`. Remove the credentials only after the call context API is
confirmed in production:

1. Deploy this AI change (no DB variables are read any more).
2. Complete the VOICE and VIDEO smoke checks above and confirm
   `[SIGNALING] CALL_ACCEPT sent` without any MySQL error.
3. On the GPU face worker, set `FACE_TRAINING_PREVIEW_FALLBACK_VOICE_ID` or
   `FACE_TRAINING_PREVIEW_FALLBACK_AUDIO_PATH` if
   `FACE_TRAINING_MEMBER_VOICE_PREVIEW_ENABLE=true`; otherwise the preview is
   skipped and logged as `member voice face preview skipped`.
4. Remove `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USERNAME`, and `DB_PASSWORD`
   from every AI server `.env` (API, call server, voice worker, GPU worker) and
   restart the services.
5. Revoke the AI server's MySQL user or security-group rule on the backend side.

Optional cache and retry settings for the call server:

```env
BACKEND_CALL_CONTEXT_MAX_ATTEMPTS=2
BACKEND_CALL_CONTEXT_RETRY_BACKOFF_SECONDS=0.2
CALL_CONTEXT_CACHE_TTL_SECONDS=600
CALL_CONTEXT_CACHE_MAX_ENTRIES=500
```
