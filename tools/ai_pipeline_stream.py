from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading

try:
    from tools.ai_pipeline_monitor import (
        ANSI_BLUE,
        ANSI_BOLD,
        ANSI_BRIGHT_CYAN,
        ANSI_DIM,
        ANSI_MAGENTA,
        ANSI_RED,
        ANSI_RESET,
        _event_line,
    )
except ModuleNotFoundError:
    from ai_pipeline_monitor import (
        ANSI_BLUE,
        ANSI_BOLD,
        ANSI_BRIGHT_CYAN,
        ANSI_DIM,
        ANSI_MAGENTA,
        ANSI_RED,
        ANSI_RESET,
        _event_line,
    )


def _paint(text: str, style: str, enabled: bool) -> str:
    return f"{style}{text}{ANSI_RESET}" if enabled else text


def format_event(source: str, line: str, *, color: bool) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S")
    source_style = ANSI_MAGENTA if source == "AI" else ANSI_BLUE
    return (
        f"{_paint(timestamp, ANSI_DIM, color)} "
        f"{_paint(f'[{source:<3}]', ANSI_BOLD + source_style, color)} "
        f"{_event_line(line, color)}"
    )


def _ssh_args(
    *, host: str, user: str, key: Path, port: int, command: str
) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
        f"{user}@{host}",
        command,
    ]


def _ai_command(history: int, scan_lines: int) -> str:
    marker = r"\[(RAG_PROFILE|VOICE_TRAINING|CLONE_SIMILARITY)\]"
    history_command = ""
    if history:
        history_command = (
            "journalctl -u mirrorsoul-ai.service -u mirrorsoul-voice-worker.service "
            f"-n {scan_lines} --no-pager -o cat 2>/dev/null "
            f"| grep -E '{marker}' | tail -n {history}; "
        )
    return (
        "{ "
        + history_command
        + "journalctl -u mirrorsoul-ai.service -u mirrorsoul-voice-worker.service "
        f"-f -n 0 -o cat 2>/dev/null | grep --line-buffered -E '{marker}'; "
        "}"
    )


def _gpu_command(face_log: str, history: int, scan_lines: int) -> str:
    marker = r"\[FACE_(TRAINING|SIMILARITY)\]"
    history_command = ""
    if history:
        history_command = (
            f"tail -n {scan_lines} {face_log} 2>/dev/null "
            f"| grep -E '{marker}' | tail -n {history}; "
        )
    return (
        "{ "
        + history_command
        + f"tail -n 0 -F {face_log} 2>/dev/null "
        f"| grep --line-buffered -E '{marker}'; "
        "}"
    )


class StreamWorker:
    def __init__(
        self,
        *,
        source: str,
        args: list[str],
        output: queue.Queue[tuple[str, str]],
        stop: threading.Event,
    ) -> None:
        self.source = source
        self.args = args
        self.output = output
        self.stop = stop
        self.process: subprocess.Popen[str] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def terminate(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                self.process = subprocess.Popen(
                    self.args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert self.process.stdout is not None
                for raw_line in self.process.stdout:
                    if self.stop.is_set():
                        break
                    line = raw_line.rstrip("\r\n")
                    if line:
                        self.output.put((self.source, line))
                return_code = self.process.wait()
                if not self.stop.is_set():
                    self.output.put(
                        (
                            "SYSTEM",
                            f"{self.source} stream disconnected (exit={return_code}); reconnecting...",
                        )
                    )
            except OSError as exc:
                self.output.put(
                    ("SYSTEM", f"{self.source} stream failed: {exc}; reconnecting...")
                )
            self.stop.wait(3)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Continuously append AI training events for every member."
    )
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--scan-lines", type=int, default=800)
    parser.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto"
    )
    parser.add_argument("--ai-host", default="13.209.220.154")
    parser.add_argument("--ai-user", default="ec2-user")
    parser.add_argument("--ai-key", type=Path, default=repo_root / "mirrorsoul-ai-key.pem")
    parser.add_argument("--gpu-host", default="203.249.75.55")
    parser.add_argument("--gpu-user", default="mirrorsoul")
    parser.add_argument("--gpu-port", type=int, default=40053)
    parser.add_argument(
        "--gpu-key",
        type=Path,
        default=Path.home() / ".ssh" / "mirrorsoul_gpu_vscode_ed25519",
    )
    parser.add_argument(
        "--face-log", default="/shareHost/C084003-ai/logs/face-worker.log"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not shutil.which("ssh"):
        print("ERROR: OpenSSH client (ssh) was not found.", file=sys.stderr)
        return 2
    if args.history < 0 or args.scan_lines <= 0:
        print("ERROR: history must be non-negative and scan-lines must be positive.")
        return 2

    color = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    events: queue.Queue[tuple[str, str]] = queue.Queue()
    stop = threading.Event()
    workers = [
        StreamWorker(
            source="AI",
            args=_ssh_args(
                host=args.ai_host,
                user=args.ai_user,
                key=args.ai_key,
                port=22,
                command=_ai_command(args.history, args.scan_lines),
            ),
            output=events,
            stop=stop,
        ),
        StreamWorker(
            source="GPU",
            args=_ssh_args(
                host=args.gpu_host,
                user=args.gpu_user,
                key=args.gpu_key,
                port=args.gpu_port,
                command=_gpu_command(args.face_log, args.history, args.scan_lines),
            ),
            output=events,
            stop=stop,
        ),
    ]

    print(
        _paint(
            "MIRROR SOUL - LIVE AI TRAINING EVENTS",
            ANSI_BOLD + ANSI_BRIGHT_CYAN,
            color,
        )
    )
    print("All members are shown in one continuous log. Ctrl+C to stop.")
    print("=" * 90, flush=True)
    for worker in workers:
        worker.start()

    try:
        while True:
            try:
                source, line = events.get(timeout=1)
            except queue.Empty:
                continue
            if source == "SYSTEM":
                print(
                    _paint(
                        f"{datetime.now():%H:%M:%S} [SYSTEM] {line}",
                        ANSI_RED,
                        color,
                    ),
                    flush=True,
                )
                continue
            print(format_event(source, line, color=color), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for worker in workers:
            worker.terminate()
    print("\nLive AI event stream stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
