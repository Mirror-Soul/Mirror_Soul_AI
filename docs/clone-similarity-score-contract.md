# Clone similarity score contract

`clones.sync_rate` is the user-facing clone completeness score. It is not a
biometric probability. Calculation version `clone-similarity-v1` combines four
independent components and an optional low-quality-data penalty.

```text
raw score = face rendering quality * 0.30
          + voice similarity * 0.30
          + personality and memory fidelity * 0.30
          + data reliability * 0.10
          - penalty

display score = round_half_up(clamp(raw score * 0.95, 0, 95), 1)
```

Missing components contribute zero and the remaining weights are not
re-normalized. This prevents an incomplete clone from appearing fully trained.
The display score has one decimal place and can never exceed `95.0`.

## Components

- Face: a fixed-audio Ditto preview evaluated for source appearance
  preservation, face detection coverage, temporal stability, geometry
  stability, and sharpness retention. It does not claim biometric identity.
- Voice: speaker similarity when the embedding evaluator succeeds, otherwise
  the existing voice-training readiness proxy.
- Profile: onboarding interview coverage, basic profile completeness, and
  accumulated user conversation memory.
- Data reliability: voice-training completion, text/audio coverage, profile
  coverage, and data maturity.
- Penalty: explicit low-quality-data deductions. Repeated, empty, or verified
  irrelevant content may add a penalty, but ordinary short answers, typos, or
  changed preferences must not be penalized automatically.

## Face completion event

The face result event includes a component contract. The backend stores the
component and recalculates the aggregate with the latest values.

```json
{
  "calculationVersion": "clone-similarity-v1",
  "faceScore": 86.5,
  "weights": {
    "face": 0.3,
    "voice": 0.3,
    "profile": 0.3,
    "dataReliability": 0.1
  },
  "displayScale": 0.95,
  "maximumScore": 95.0,
  "confidence": "high",
  "calibrationVersion": "member-v1",
  "calibrated": true
}
```

## Backend migration

The backend owns the schema migration and final aggregate. Keep `sync_rate` for
the existing API and store every component separately so asynchronous jobs
cannot overwrite unrelated values.

After RAG profile training, the AI completion callback includes the components
owned by that flow:

```http
POST /internal/clone-training/{cloneId}/personality/complete
Content-Type: application/json

{
  "calculationVersion": "clone-similarity-v1",
  "profileScore": 64.25,
  "dataReliabilityScore": 91.5,
  "penaltyScore": 1.5
}
```

For rollout compatibility the callback helper can still send the old empty
body when no score components are available. The backend should accept both
forms until every AI server is updated.

```sql
ALTER TABLE clones
    ADD COLUMN face_similarity_score DECIMAL(5,2) NULL,
    ADD COLUMN voice_similarity_score DECIMAL(5,2) NULL,
    ADD COLUMN profile_similarity_score DECIMAL(5,2) NULL,
    ADD COLUMN data_reliability_score DECIMAL(5,2) NULL,
    ADD COLUMN similarity_penalty DECIMAL(5,2) NOT NULL DEFAULT 0,
    ADD COLUMN similarity_score_version VARCHAR(50) NULL;
```

Do not copy the legacy `sync_rate` into `voice_similarity_score`: the legacy
value mixed voice readiness, interview coverage, and profile completeness. Keep
the old display value until the member's components are recalculated, then set
`similarity_score_version` and replace `sync_rate` atomically.

On every face, voice, or RAG-profile update, the backend must lock the clone
row, update only the component owned by that event, and recalculate `sync_rate`
in the same transaction. Store `clone-similarity-v1` in
`similarity_score_version`.

Late, duplicate, or older events must not replace newer active component data.
The API returns `sync_rate` as a JSON number with one decimal place, for example
`60.7` or `94.7`.

The AI voice worker remains compatible before the migration. MySQL error 1054
for missing component columns triggers the legacy score write, while the new
component aggregate remains inactive until the backend schema is ready.
