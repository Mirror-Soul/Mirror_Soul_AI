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
