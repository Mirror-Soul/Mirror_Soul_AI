import asyncio
import base64
import unittest
from types import SimpleNamespace

from model_calling.realtime.transcription import (
    RealtimeTranscriber,
    RealtimeTranscriptionError,
)


class _InputBuffer:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.appended: list[str] = []
        self.cleared = 0
        self.commits = 0

    async def append(self, *, audio: str) -> None:
        self.appended.append(audio)

    async def clear(self) -> None:
        self.cleared += 1

    async def commit(self) -> None:
        self.commits += 1
        item_id = f"item-{self.commits}"
        await self.connection.events.put(
            SimpleNamespace(type="input_audio_buffer.committed", item_id=item_id)
        )
        await self.connection.events.put(
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                item_id=item_id,
                transcript=" 안녕하세요 ",
            )
        )


class _Connection:
    def __init__(self) -> None:
        self.events: asyncio.Queue = asyncio.Queue()
        self.input_audio_buffer = _InputBuffer(self)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.events.get()


class _Manager:
    async def __aexit__(self, *args) -> None:
        return None


class RealtimeTranscriberTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.connection = _Connection()
        self.transcriber = RealtimeTranscriber(_Manager(), self.connection)
        self.transcriber._reader_task = asyncio.create_task(
            self.transcriber._read_events()
        )

    async def asyncTearDown(self) -> None:
        await self.transcriber.close()

    async def test_commit_flushes_audio_and_resolves_transcript(self) -> None:
        pcm = b"\x01\x02" * 1_000
        await self.transcriber.append_audio(pcm)
        transcript = await (await self.transcriber.commit())

        self.assertEqual(transcript, "안녕하세요")
        sent = b"".join(
            base64.b64decode(chunk)
            for chunk in self.connection.input_audio_buffer.appended
        )
        self.assertEqual(sent, pcm)
        self.assertEqual(self.connection.input_audio_buffer.commits, 1)

    async def test_clear_discards_unsent_audio(self) -> None:
        await self.transcriber.append_audio(b"noise")
        await self.transcriber.clear()
        self.assertEqual(self.connection.input_audio_buffer.cleared, 1)
        self.assertEqual(self.transcriber._audio_buffer, bytearray())

    async def test_connection_error_fails_pending_result(self) -> None:
        future = asyncio.get_running_loop().create_future()
        self.transcriber._items["item"] = future
        self.transcriber._fail_all(RealtimeTranscriptionError("down"))
        with self.assertRaises(RealtimeTranscriptionError):
            await future


if __name__ == "__main__":
    unittest.main()
