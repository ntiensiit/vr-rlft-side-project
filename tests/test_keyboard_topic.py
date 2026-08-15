"""Tests for UDP keyboard topic teleoperation."""

from __future__ import annotations

import json
import socket

from grasping_ai.pipelines.visualize_robot import run_keyboard_tui


def _bind_topic_socket() -> socket.socket:
    """Bind a UDP socket to local loopback on an ephemeral port for testing."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    return sock


def _drain_topic_keycodes(sock: socket.socket) -> list[int]:
    """Receive and parse keycode messages from a bound socket."""
    topic = "robot/keyboard"
    keycodes: list[int] = []
    while True:
        try:
            payload, _addr = sock.recvfrom(4096)
        except BlockingIOError:
            break
        except OSError:
            break
        try:
            message = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(message, dict):
            continue
        if message.get("topic") != topic:
            continue
        keycode = message.get("keycode")
        if isinstance(keycode, int):
            keycodes.append(keycode)
    return keycodes


def test_keyboard_topic_round_trip() -> None:
    """Verify that a keyboard topic message sent over UDP is correctly received and parsed."""
    topic_sock = _bind_topic_socket()
    port = int(topic_sock.getsockname()[1])
    pub_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        pub_sock.sendto(
            b'{"topic":"robot/keyboard","keycode":71}',
            ("127.0.0.1", port),
        )
        received: list[int] = []
        for _ in range(50):
            received.extend(_drain_topic_keycodes(topic_sock))
            if received:
                break
        assert received == [71]
    finally:
        pub_sock.close()
        topic_sock.close()


def test_run_keyboard_tui_publishes_then_quits() -> None:
    """Verify that the keyboard TUI publishes parsed keycodes and terminates on exit indicator."""
    topic_sock = _bind_topic_socket()
    port = int(topic_sock.getsockname()[1])
    keys = [72, -1]
    try:
        run_keyboard_tui(
            host="127.0.0.1",
            port=port,
            read_key=lambda: keys.pop(0),
            sleep=lambda _dt: None,
        )
        assert _drain_topic_keycodes(topic_sock) == [72]
    finally:
        topic_sock.close()
