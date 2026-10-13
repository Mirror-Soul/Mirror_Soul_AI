import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from model_calling import services
from model_calling.realtime.prewarm import prewarm_realtime_dependencies
from shared.config import settings


class RealtimeDependencyPrewarmTests(unittest.TestCase):
    def test_openai_warmup_only_retrieves_model_metadata(self) -> None:
        retrieve = AsyncMock(return_value=object())
        fake_client = SimpleNamespace(
            models=SimpleNamespace(retrieve=retrieve),
            chat=Mock(),
            audio=Mock(),
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(settings, "REALTIME_LLM_MODEL", "gpt-4o-mini"),
            patch.object(services, "client", fake_client),
        ):
            model = asyncio.run(services.warmup_openai_connection())

        self.assertEqual(model, "gpt-4o-mini")
        retrieve.assert_awaited_once_with("gpt-4o-mini")
        self.assertEqual(fake_client.chat.mock_calls, [])
        self.assertEqual(fake_client.audio.mock_calls, [])

    def test_dependencies_are_prepared_in_parallel(self) -> None:
        async def run():
            active = 0
            peak = 0

            async def warmup(*args, **kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

            with (
                patch.object(
                    settings,
                    "REALTIME_DEPENDENCY_PREWARM_ENABLED",
                    True,
                ),
                patch.object(
                    settings,
                    "REALTIME_DEPENDENCY_PREWARM_TIMEOUT_SECONDS",
                    1.0,
                ),
                patch(
                    "model_calling.realtime.prewarm.warmup_openai_connection",
                    new=AsyncMock(side_effect=warmup),
                ),
                patch(
                    "model_calling.realtime.prewarm.warmup_elevenlabs_connection",
                    new=AsyncMock(side_effect=warmup),
                ),
                patch(
                    "model_calling.realtime.prewarm.warmup_rag_connection",
                    new=Mock(side_effect=lambda: asyncio.run(warmup())),
                ),
            ):
                statuses = await prewarm_realtime_dependencies(51)
            return statuses, peak

        statuses, peak = asyncio.run(run())

        self.assertEqual(
            statuses,
            {"openai": "ready", "rag": "ready", "elevenlabs": "ready"},
        )
        self.assertEqual(peak, 3)

    def test_failure_is_advisory_and_other_dependencies_still_finish(self) -> None:
        with (
            patch.object(
                settings,
                "REALTIME_DEPENDENCY_PREWARM_ENABLED",
                True,
            ),
            patch.object(
                settings,
                "REALTIME_DEPENDENCY_PREWARM_TIMEOUT_SECONDS",
                1.0,
            ),
            patch(
                "model_calling.realtime.prewarm.warmup_openai_connection",
                new=AsyncMock(side_effect=RuntimeError("unavailable")),
            ),
            patch(
                "model_calling.realtime.prewarm.warmup_elevenlabs_connection",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "model_calling.realtime.prewarm.warmup_rag_connection",
                new=Mock(return_value="remote"),
            ),
        ):
            statuses = asyncio.run(prewarm_realtime_dependencies(52))

        self.assertEqual(statuses["openai"], "warning:RuntimeError")
        self.assertEqual(statuses["rag"], "ready")
        self.assertEqual(statuses["elevenlabs"], "ready")

    def test_disabled_warmup_makes_no_external_calls(self) -> None:
        openai = AsyncMock()
        with (
            patch.object(
                settings,
                "REALTIME_DEPENDENCY_PREWARM_ENABLED",
                False,
            ),
            patch(
                "model_calling.realtime.prewarm.warmup_openai_connection",
                new=openai,
            ),
        ):
            statuses = asyncio.run(prewarm_realtime_dependencies(53))

        self.assertEqual(statuses, {"prewarm": "disabled"})
        openai.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
