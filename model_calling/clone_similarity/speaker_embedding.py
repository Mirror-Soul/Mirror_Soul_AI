import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx


class SpeakerSimilarityUnavailable(Exception):
    pass


@dataclass(frozen=True)
class SpeakerSimilarityAudio:
    filename: str
    content: bytes
    content_type: str


@dataclass(frozen=True)
class SpeakerSimilarityResult:
    score: float
    cosine_similarity: float
    original_sample_count: int
    model_name: str
    reference_audio_path: str | None = None


_CLASSIFIER = None
DEFAULT_REFERENCE_TEXT = (
    "\uc548\ub155\ud558\uc138\uc694! "
    "\ucc98\uc74c\ubd59\uaca0\uc2b5\ub2c8\ub2e4."
)


def evaluate_speaker_similarity(
    *,
    original_audios: Sequence[SpeakerSimilarityAudio],
    elevenlabs_voice_id: str,
    reference_audio_path: str | Path | None = None,
) -> SpeakerSimilarityResult:
    if not _env_bool("CLONE_SIMILARITY_ENABLE_SPEAKER_EMBEDDING", False):
        raise SpeakerSimilarityUnavailable("speaker embedding evaluation is disabled")
    if not original_audios:
        raise SpeakerSimilarityUnavailable("no original audio samples provided")

    model_name = os.getenv(
        "CLONE_SIMILARITY_SPEAKER_MODEL",
        "speechbrain/spkrec-ecapa-voxceleb",
    )

    with tempfile.TemporaryDirectory(prefix="mirror-soul-speaker-") as temp_dir:
        temp_path = Path(temp_dir)
        original_paths = [
            _write_audio_file(temp_path, index, audio)
            for index, audio in enumerate(original_audios, start=1)
        ]
        if reference_audio_path:
            clone_path = Path(reference_audio_path)
            if not clone_path.exists():
                clone_path = generate_clone_reference_audio_file(
                    elevenlabs_voice_id=elevenlabs_voice_id,
                    output_path=reference_audio_path,
                )
        else:
            clone_path = generate_clone_reference_audio_file(
                elevenlabs_voice_id=elevenlabs_voice_id,
                output_path=temp_path / "clone-reference.mp3",
            )

        classifier = _get_classifier(model_name)
        original_embedding = _average_embeddings(classifier, original_paths)
        clone_embedding = _encode_file(classifier, clone_path)
        cosine_similarity = _cosine_similarity(original_embedding, clone_embedding)

    return SpeakerSimilarityResult(
        score=_cosine_to_score(cosine_similarity),
        cosine_similarity=round(cosine_similarity, 4),
        original_sample_count=len(original_audios),
        model_name=model_name,
        reference_audio_path=str(clone_path) if reference_audio_path else None,
    )


def generate_clone_reference_audio_file(
    *,
    elevenlabs_voice_id: str,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_generate_clone_reference_audio(elevenlabs_voice_id))
    return path


def _generate_clone_reference_audio(elevenlabs_voice_id: str) -> bytes:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SpeakerSimilarityUnavailable("ELEVENLABS_API_KEY is not configured")

    text = os.getenv(
        "CLONE_SIMILARITY_REFERENCE_TEXT",
        DEFAULT_REFERENCE_TEXT,
    )
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }
    payload = {
        "text": text,
        "model_id": os.getenv("ELEVENLABS_TTS_MODEL_ID", "eleven_multilingual_v2"),
        "voice_settings": {
            "stability": _env_float("CLONE_SIMILARITY_TTS_STABILITY", 0.55),
            "similarity_boost": _env_float("CLONE_SIMILARITY_TTS_SIMILARITY_BOOST", 0.9),
            "style": _env_float("CLONE_SIMILARITY_TTS_STYLE", 0.0),
            "use_speaker_boost": True,
        },
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise SpeakerSimilarityUnavailable(
            f"ElevenLabs reference TTS failed [{response.status_code}]: {response.text}"
        )
    return response.content


def _get_classifier(model_name: str):
    global _CLASSIFIER
    if _CLASSIFIER is not None:
        return _CLASSIFIER

    try:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
    except ImportError as exc:
        raise SpeakerSimilarityUnavailable(
            "speechbrain is not installed. Install requirements-voice-similarity.txt."
        ) from exc

    savedir = os.getenv(
        "CLONE_SIMILARITY_SPEAKER_MODEL_DIR",
        "model_calling/assets/speaker_embedding_model",
    )
    _CLASSIFIER = EncoderClassifier.from_hparams(
        source=model_name,
        savedir=savedir,
    )
    return _CLASSIFIER


def _write_audio_file(
    temp_path: Path,
    index: int,
    audio: SpeakerSimilarityAudio,
) -> Path:
    suffix = Path(audio.filename).suffix
    if not suffix:
        suffix = _suffix_from_content_type(audio.content_type)
    path = temp_path / f"original-{index}{suffix}"
    path.write_bytes(audio.content)
    return path


def _suffix_from_content_type(content_type: str) -> str:
    normalized = content_type.lower()
    if "mpeg" in normalized or "mp3" in normalized:
        return ".mp3"
    if "m4a" in normalized or "mp4" in normalized:
        return ".m4a"
    if "webm" in normalized:
        return ".webm"
    return ".wav"


def _average_embeddings(classifier, paths: Sequence[Path]):
    embeddings = [_encode_file(classifier, path) for path in paths]
    try:
        import torch
    except ImportError as exc:
        raise SpeakerSimilarityUnavailable(
            "torch is not installed. Install requirements-voice-similarity.txt."
        ) from exc

    stacked = torch.stack(embeddings)
    return stacked.mean(dim=0)


def _encode_file(classifier, path: Path):
    if hasattr(classifier, "encode_file"):
        embedding = classifier.encode_file(str(path))
        return embedding.squeeze().detach().cpu()

    signal = _load_audio_signal(classifier, path)
    try:
        import torch
    except ImportError as exc:
        raise SpeakerSimilarityUnavailable(
            "torch is not installed. Install requirements-voice-similarity.txt."
        ) from exc

    with torch.no_grad():
        embedding = classifier.encode_batch(signal.unsqueeze(0))
    return embedding.squeeze().detach().cpu()


def _load_audio_signal(classifier, path: Path):
    if hasattr(classifier, "load_audio"):
        return classifier.load_audio(str(path)).squeeze()

    try:
        import torchaudio
    except ImportError as exc:
        raise SpeakerSimilarityUnavailable(
            "torchaudio is not installed. Install requirements-voice-similarity.txt."
        ) from exc

    signal, sample_rate = torchaudio.load(str(path))
    if signal.ndim > 1:
        signal = signal.mean(dim=0)

    target_sample_rate = int(getattr(classifier.hparams, "sample_rate", 16000))
    if sample_rate != target_sample_rate:
        signal = torchaudio.functional.resample(
            signal,
            sample_rate,
            target_sample_rate,
        )
    return signal.squeeze()


def _cosine_similarity(left, right) -> float:
    try:
        import torch.nn.functional as F
    except ImportError as exc:
        raise SpeakerSimilarityUnavailable(
            "torch is not installed. Install requirements-voice-similarity.txt."
        ) from exc

    return float(F.cosine_similarity(left, right, dim=0).item())


def _cosine_to_score(cosine_similarity: float) -> float:
    low = _env_float("CLONE_SIMILARITY_COSINE_LOW", 0.20)
    high = _env_float("CLONE_SIMILARITY_COSINE_HIGH", 0.70)
    max_score = _env_float("CLONE_SIMILARITY_MAX_ACTUAL_VOICE_SCORE", 95.0)
    if high <= low:
        low, high = 0.20, 0.70

    normalized = (cosine_similarity - low) / (high - low)
    return round(max(0.0, min(normalized, 1.0)) * max_score, 2)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default
