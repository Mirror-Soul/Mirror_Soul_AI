from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class DittoServiceConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class DittoServiceConfig:
    api_key: str
    repository_dir: Path = Path(
        "/shareHost/C084003-ditto/ditto-talkinghead"
    )
    data_root: Path | None = None
    model_config_path: Path | None = None
    ffmpeg_dir: Path = Path("/opt/conda/bin")
    host: str = "127.0.0.1"
    port: int = 8080
    max_portrait_bytes: int = 10 * 1024 * 1024
    max_audio_bytes: int = 25 * 1024 * 1024
    require_cuda: bool = True
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None

    @classmethod
    def from_env(cls) -> "DittoServiceConfig":
        api_key = os.getenv("DITTO_SERVICE_API_KEY", "").strip()
        if not api_key:
            raise DittoServiceConfigError(
                "DITTO_SERVICE_API_KEY must be configured."
            )

        return cls(
            api_key=api_key,
            repository_dir=Path(
                os.getenv(
                    "DITTO_SERVICE_REPO_DIR",
                    "/shareHost/C084003-ditto/ditto-talkinghead",
                )
            ),
            data_root=_env_path("DITTO_SERVICE_DATA_ROOT"),
            model_config_path=_env_path("DITTO_SERVICE_CONFIG_PATH"),
            ffmpeg_dir=Path(
                os.getenv("DITTO_SERVICE_FFMPEG_DIR", "/opt/conda/bin")
            ),
            host=os.getenv("DITTO_SERVICE_HOST", "127.0.0.1"),
            port=_env_int("DITTO_SERVICE_PORT", 8080),
            max_portrait_bytes=_env_int(
                "DITTO_SERVICE_MAX_PORTRAIT_BYTES",
                10 * 1024 * 1024,
            ),
            max_audio_bytes=_env_int(
                "DITTO_SERVICE_MAX_AUDIO_BYTES",
                25 * 1024 * 1024,
            ),
            require_cuda=_env_bool("DITTO_SERVICE_REQUIRE_CUDA", True),
            ssl_certfile=_env_path("DITTO_SERVICE_SSL_CERTFILE"),
            ssl_keyfile=_env_path("DITTO_SERVICE_SSL_KEYFILE"),
        )

    @property
    def resolved_data_root(self) -> Path:
        return self._resolve_from_repository(
            self.data_root,
            "checkpoints/ditto_pytorch",
        )

    @property
    def resolved_model_config_path(self) -> Path:
        return self._resolve_from_repository(
            self.model_config_path,
            "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        )

    def validate(self) -> None:
        if self.port <= 0 or self.port > 65535:
            raise DittoServiceConfigError("DITTO_SERVICE_PORT is invalid.")
        if self.max_portrait_bytes <= 0 or self.max_audio_bytes <= 0:
            raise DittoServiceConfigError(
                "Ditto upload size limits must be positive."
            )
        if (self.ssl_certfile is None) != (self.ssl_keyfile is None):
            raise DittoServiceConfigError(
                "DITTO_SERVICE_SSL_CERTFILE and DITTO_SERVICE_SSL_KEYFILE "
                "must be configured together."
            )
        for path, label in (
            (self.ssl_certfile, "Ditto TLS certificate"),
            (self.ssl_keyfile, "Ditto TLS private key"),
        ):
            if path is not None and not path.resolve().is_file():
                raise DittoServiceConfigError(f"{label} not found: {path.resolve()}")
        if not self.repository_dir.resolve().is_dir():
            raise DittoServiceConfigError(
                f"Ditto repository not found: {self.repository_dir.resolve()}"
            )
        if not self.resolved_data_root.is_dir():
            raise DittoServiceConfigError(
                f"Ditto checkpoint directory not found: {self.resolved_data_root}"
            )
        if not self.resolved_model_config_path.is_file():
            raise DittoServiceConfigError(
                f"Ditto config not found: {self.resolved_model_config_path}"
            )
        if not self.ffmpeg_dir.resolve().is_dir():
            raise DittoServiceConfigError(
                f"Ditto ffmpeg directory not found: {self.ffmpeg_dir.resolve()}"
            )

    def _resolve_from_repository(
        self,
        configured_path: Path | None,
        default_relative_path: str,
    ) -> Path:
        path = configured_path or Path(default_relative_path)
        if path.is_absolute():
            return path.resolve()
        return (self.repository_dir.resolve() / path).resolve()


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value else default
    except ValueError as exc:
        raise DittoServiceConfigError(f"{name} must be an integer.") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DittoServiceConfigError(f"{name} must be a boolean value.")
