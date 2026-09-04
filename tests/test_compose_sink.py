"""The compose sink: dictation goes to an open compose window, or nowhere special."""

import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from vocalinux.text_injection import compose_sink


class _Window:
    """A stand-in compose window: one accept loop, scripted replies."""

    def __init__(self, path: Path, reply=b'{"ok": true}\n'):
        self.path = path
        self.reply = reply
        self.received = []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(path))
        self.sock.listen(4)
        self.sock.settimeout(2.0)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except (OSError, socket.timeout):
                return
            with conn:
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                self.received.append(json.loads(data.decode()))
                if self.reply is not None:
                    conn.sendall(self.reply)

    def close(self):
        self.sock.close()
        self.thread.join(timeout=2.0)


class TestComposeSink(unittest.TestCase):
    def setUp(self):
        # A Unix socket path is capped at ~108 bytes; a plain tempdir under
        # /tmp is short enough, a pytest tmp_path often is not.
        self.tmp = tempfile.TemporaryDirectory(prefix="vcs-")
        self.path = Path(self.tmp.name) / "c.sock"
        self.env = patch.dict(os.environ, {compose_sink.SOCKET_ENV: str(self.path)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_no_window_means_not_delivered(self):
        self.assertFalse(compose_sink.deliver("hello", new_session=True))

    def test_open_window_takes_the_segment(self):
        window = _Window(self.path)
        try:
            self.assertTrue(compose_sink.deliver("hello there", new_session=True))
            self.assertTrue(compose_sink.deliver("and more", new_session=False))
        finally:
            window.close()
        self.assertEqual(
            window.received,
            [
                {"text": "hello there", "new_session": True},
                {"text": "and more", "new_session": False},
            ],
        )

    def test_stale_socket_file_is_no_window(self):
        window = _Window(self.path)
        window.close()  # the file stays behind, nothing listens
        self.assertTrue(self.path.exists())
        self.assertFalse(compose_sink.deliver("hello", new_session=True))

    def test_window_that_refuses_is_not_delivered(self):
        window = _Window(self.path, reply=b'{"ok": false, "why": "finishing"}\n')
        try:
            self.assertFalse(compose_sink.deliver("hello", new_session=True))
        finally:
            window.close()

    def test_window_that_hangs_up_is_not_delivered(self):
        window = _Window(self.path, reply=None)
        try:
            self.assertFalse(compose_sink.deliver("hello", new_session=True))
        finally:
            window.close()

    def test_garbage_reply_is_not_delivered(self):
        window = _Window(self.path, reply=b"ok\n")
        try:
            self.assertFalse(compose_sink.deliver("hello", new_session=True))
        finally:
            window.close()

    def test_default_path_is_under_the_runtime_dir(self):
        with patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/1234"}, clear=False):
            os.environ.pop(compose_sink.SOCKET_ENV, None)
            self.assertEqual(
                compose_sink.socket_path(), Path("/run/user/1234/vocalinux/compose.sock")
            )


if __name__ == "__main__":
    unittest.main()
