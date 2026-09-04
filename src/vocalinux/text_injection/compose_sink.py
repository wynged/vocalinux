"""Hand dictation to a compose window instead of the focused app, while one is open.

Dictating a long prompt in several holds of the key has two problems: the
focused window wanders (you read a browser tab, scroll some code, click a
thing) and each hold lands wherever focus happens to be; and with no text
field focused at all the words have nowhere to go. A compose window is a
place they always go. It is a separate process -- ``vocalinux-compose``, a
small curses program in a floating terminal -- that listens on a Unix socket
for as long as it is open. Every finalised segment is offered to that socket
first; when the window takes it, nothing is pasted or typed anywhere else.

The window's lifetime IS the mode. There is no flag to set or forget: the
socket exists while the window runs and is gone when it exits, and the next
segment goes back to the focused field. A socket file left behind by a
window that was killed refuses the connection, which reads the same as no
window at all.

Protocol, one JSON object per line, one connection per segment::

    -> {"text": "the segment", "new_session": true}
    <- {"ok": true}

``new_session`` is true for the first segment after the dictation key was
pressed anew (IDLE -> LISTENING), so the window can start a paragraph; later
segments of the same hold are joined with a space, as they would be in a
text field. The text is sent bare, without the inter-segment space the
injector would add, because the window decides how to join.

The check costs one ``stat`` per segment while no window is open, and one
short-lived connection while one is. Anything that goes wrong -- a missing
socket, a refused connection, a reply that is not ``ok`` -- falls back to
normal injection, so a broken window can never eat dictation silently.
"""

import json
import logging
import os
import socket
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The environment override is for tests and for a second, experimental window;
# ``vocalinux-compose`` honours the same variable.
SOCKET_ENV = "VOCALINUX_COMPOSE_SOCKET"
REPLY_TIMEOUT = 2.0
MAX_REPLY = 4096


def socket_path() -> Path:
    """Where a compose window listens. Shared with ``vocalinux-compose``."""
    override = os.environ.get(SOCKET_ENV)
    if override:
        return Path(override)
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(base) / "vocalinux" / "compose.sock"


def deliver(text: str, new_session: bool) -> bool:
    """Offer one segment to the compose window. True if it took the text."""
    if not hasattr(socket, "AF_UNIX"):
        return False
    path = socket_path()
    try:
        if not path.exists():
            return False
    except OSError:
        return False
    request = json.dumps({"text": text, "new_session": bool(new_session)}) + "\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(REPLY_TIMEOUT)
            sock.connect(str(path))
            sock.sendall(request.encode("utf-8"))
            reply = _read_line(sock)
    except OSError as e:
        # ENOENT/ECONNREFUSED: no window (or a stale socket file). A timeout:
        # a window that stopped answering. Either way the text goes to the
        # focused field instead, so nothing is lost.
        logger.debug(f"Compose window not reachable at {path}: {e}")
        return False
    accepted = _accepted(reply)
    if accepted:
        logger.info(f"Compose window took the segment ({len(text)} chars)")
    else:
        logger.warning(f"Compose window refused the segment: {reply!r}")
    return accepted


def _read_line(sock: socket.socket) -> Optional[bytes]:
    buf = b""
    while b"\n" not in buf and len(buf) < MAX_REPLY:
        chunk = sock.recv(1024)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0] if buf else None


def _accepted(reply: Optional[bytes]) -> bool:
    if not reply:
        return False
    try:
        return json.loads(reply.decode("utf-8")).get("ok") is True
    except (ValueError, AttributeError):
        return False
