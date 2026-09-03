"""In-process ownership of the X11 CLIPBOARD selection, with read detection.

``xclip`` can put text on the clipboard, but it cannot answer the question
dictation actually needs: did anything *read* it? X11 hands a selection over
on request, so the owner of the selection is told -- with the requesting
window's id -- every time a client converts it. Holding the clipboard here is
what turns "we pressed Ctrl+V" into "the focused app took the text", and so
what lets a paste with nowhere to land be reported instead of silently lost.

Measured on an i3/X11 desktop: a paste into a GTK entry, into Chrome's
omnibox and into a WezTerm pane each converted ``UTF8_STRING`` within ~30 ms
of the keystroke. A Ctrl+V into Chrome with no field focused converted
nothing -- it asked only for ``TARGETS``, which is the toolkit deciding
whether to enable a Paste menu item, not a paste -- and a Ctrl+V into an
Xlib app with no text handling asked for nothing at all. So a *content*
target is the signal and ``TARGETS`` is not.

One complication: a clipboard manager (greenclip here) converts the selection
roughly once a second whether or not anyone pastes, so "somebody read it" is
not the question to ask. The question is whether the *focused* application
read it, and X answers that too, because a paste is requested by a window
belonging to the same X client as the focused window -- Xorg gives each
client a 2**21 block of resource ids, so the two ids share a base. Chrome
asks through a hidden window it calls "Chromium clipboard"; a GTK dialog and
a WezTerm pane each asked through a window of their own client. Nothing else
counts, deliberately: trusting an unrecognised requestor let the very first
dictation of a session pass because the clipboard manager's poll happened to
fall inside the window.

Known limitation: Chromium reads the clipboard on Ctrl+V whether or not
anything editable is focused, because it has to build the DOM paste event
either way. So a paste into a browser page always reads as landing, and the
check has nothing to say about Chrome and Electron windows.
"""

import logging
import os
import queue
import select
import threading
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# Xorg allocates resource ids to each client in blocks of 2**21, so two window
# ids belonging to one client share every bit above the low 21. A paste is
# requested by the focused window itself or by a hidden helper window of the
# same client (Chrome's is called "Chromium clipboard"), never by another.
CLIENT_ID_MASK = ~0x1FFFFF


def client_of(window_id: Optional[int]) -> Optional[int]:
    """The X client a window id belongs to, or None if there is no window."""
    if not window_id:
        return None
    return window_id & CLIENT_ID_MASK


class _Watch:
    """A detection window opened around a paste keystroke."""

    def __init__(self, owner: "ClipboardOwner", mark: int):
        self._owner = owner
        self.mark = mark

    def landed(self, client_id: Optional[int], timeout: float) -> bool:
        """True once the focused app read the clipboard; False if it never did.

        Returns as soon as that read arrives, so a successful paste costs the
        ~30 ms the app takes to ask, and only a failure waits out ``timeout``.
        A read by anyone else is not an answer: a clipboard manager converts
        the selection about once a second whatever the user is doing, and with
        no focused client to match against (``client_id`` None) there is
        nothing that could have taken the text.
        """
        return self._owner._wait_for_read(self.mark, client_id, timeout)


class ClipboardOwner:
    """Owns the CLIPBOARD selection and records who converts it.

    Every X call happens on one background thread -- python-xlib is not
    thread-safe, and the thread has to keep serving conversion requests while
    the caller waits for one. Public methods hand work to that thread through
    a queue and a self-pipe, and read results under a lock.
    """

    # Serving a selection in one shot is capped by the server's maximum
    # request size; the alternative (the INCR protocol) is a lot of machinery
    # for text no dictation produces. Anything larger falls back to xclip.
    _SIZE_MARGIN = 1024

    def __init__(self):
        self._cond = threading.Condition()
        self._text: Optional[str] = None
        self._reads: list = []  # (seq, requestor_id) per content conversion
        self._read_seq = 0  # marks index this, not the list, so trimming is safe
        self._owns = False
        self._max_bytes = 0

        self._display = None
        self._window = None
        self._thread: Optional[threading.Thread] = None
        self._commands: "queue.Queue" = queue.Queue()
        self._wake_r = -1
        self._wake_w = -1
        self._started = False
        self._unavailable = False

    # --- lifecycle ------------------------------------------------------

    def start(self) -> bool:
        """Connect to X and run the selection thread. Idempotent."""
        with self._cond:
            if self._started:
                return True
            if self._unavailable:
                return False
            if not os.environ.get("DISPLAY"):
                self._unavailable = True
                return False
            try:
                self._connect()
            # Deliberately broad: this is an optional upgrade over xclip. Any
            # failure to reach X must leave the caller on the old path, never
            # raise into an injection.
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Clipboard owner unavailable: {e}")
                self._unavailable = True
                return False
            self._started = True
        return True

    def _connect(self) -> None:
        from Xlib import X, Xatom, display
        from Xlib.protocol import event as xevent

        self._x = X
        self._xatom = Xatom
        self._xevent = xevent
        d = display.Display()
        self._display = d
        self._window = d.screen().root.create_window(0, 0, 1, 1, 0, X.CopyFromParent)

        atom = d.get_atom
        self._a_clipboard = atom("CLIPBOARD")
        self._a_targets = atom("TARGETS")
        self._a_timestamp = atom("TIMESTAMP")
        self._a_utf8 = atom("UTF8_STRING")
        # Chrome and GTK both ask for UTF8_STRING in practice; the MIME spellings
        # and the legacy STRING/TEXT atoms are cheap to answer and keep older or
        # pickier toolkits from falling through to "no text available".
        self._content_targets = {
            self._a_utf8,
            atom("STRING"),
            atom("TEXT"),
            atom("text/plain"),
            atom("text/plain;charset=utf-8"),
        }
        self._offered_targets = [
            self._a_targets,
            self._a_timestamp,
            self._a_utf8,
            atom("STRING"),
            atom("TEXT"),
        ]
        self._max_bytes = max(
            65536, d.display.info.max_request_length * 4 - self._SIZE_MARGIN
        )

        # An async BadWindow is routine here: a requestor can go away between
        # asking and being answered. Swallow it rather than let python-xlib
        # raise it out of the event loop.
        d.set_error_handler(lambda err, req: logger.debug(f"X error (ignored): {err}"))

        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        self._thread = threading.Thread(
            target=self._run, name="clipboard-owner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._post(("stop",))
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

    # --- public API -----------------------------------------------------

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def owns(self) -> bool:
        with self._cond:
            return self._owns

    def current_text(self) -> Optional[str]:
        """What we are serving, if we still hold the selection."""
        with self._cond:
            return self._text if self._owns else None

    def set_text(self, text: str, timeout: float = 1.0) -> bool:
        """Take ownership of the clipboard and serve ``text`` from now on."""
        if not self.start():
            return False
        if len(text.encode("utf-8")) > self._max_bytes:
            logger.debug("Text too large to serve in one selection request")
            return False
        done = threading.Event()
        result = []
        self._post(("set", text, done, result))
        if not done.wait(timeout):
            logger.warning("Clipboard owner did not take ownership in time")
            return False
        return bool(result and result[0])

    @contextmanager
    def watch(self):
        """Open a detection window around a paste, ignoring every earlier read."""
        with self._cond:
            mark = self._read_seq
        yield _Watch(self, mark)

    def _wait_for_read(self, mark: int, client_id: Optional[int], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for seq, rid in self._reads:
                    if seq < mark:
                        continue
                    if client_id is not None and (rid & CLIENT_ID_MASK) == client_id:
                        return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)

    # --- the X thread ---------------------------------------------------

    def _post(self, command) -> None:
        self._commands.put(command)
        try:
            os.write(self._wake_w, b"\x01")
        except OSError:
            pass

    def _run(self) -> None:
        d = self._display
        xfd = d.fileno()
        try:
            while True:
                select.select([xfd, self._wake_r], [], [])
                try:
                    os.read(self._wake_r, 4096)
                except OSError:
                    pass
                if not self._drain_commands():
                    return
                while d.pending_events():
                    self._handle(d.next_event())
        # Deliberately broad: the thread dying quietly would leave set_text()
        # timing out on every injection with no explanation in the log.
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Clipboard owner thread stopped: {e}", exc_info=True)
        finally:
            with self._cond:
                self._owns = False
                self._started = False
                self._unavailable = True

    def _drain_commands(self) -> bool:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return True
            if command[0] == "stop":
                return False
            if command[0] == "set":
                _, text, done, result = command
                try:
                    result.append(self._take_ownership(text))
                except Exception as e:  # noqa: BLE001 - report, do not kill the thread
                    logger.warning(f"Could not take clipboard ownership: {e}")
                    result.append(False)
                finally:
                    done.set()

    def _take_ownership(self, text: str) -> bool:
        d = self._display
        self._window.set_selection_owner(self._a_clipboard, self._x.CurrentTime)
        d.sync()
        got = d.get_selection_owner(self._a_clipboard).id == self._window.id
        with self._cond:
            self._text = text
            self._owns = got
        return got

    def _handle(self, ev) -> None:
        X = self._x
        if ev.type == X.SelectionClear:
            with self._cond:
                self._owns = False
                self._cond.notify_all()
            return
        if ev.type != X.SelectionRequest:
            return

        Xatom, xevent = self._xatom, self._xevent
        with self._cond:
            text = self._text
        target = ev.target
        # A pre-ICCCM requestor sends property None and means "use the target".
        prop = ev.property if ev.property != X.NONE else target
        served = True
        content = False
        try:
            if target == self._a_targets:
                ev.requestor.change_property(prop, Xatom.ATOM, 32, self._offered_targets)
            elif target == self._a_timestamp:
                ev.requestor.change_property(prop, Xatom.INTEGER, 32, [int(ev.time)])
            elif target in self._content_targets and text is not None:
                ev.requestor.change_property(prop, target, 8, text.encode("utf-8"))
                content = True
            else:
                served = False
            ev.requestor.send_event(
                xevent.SelectionNotify(
                    time=ev.time,
                    requestor=ev.requestor,
                    selection=ev.selection,
                    target=target,
                    property=prop if served else X.NONE,
                ),
                event_mask=X.NoEventMask,
            )
            self._display.flush()
        except Exception as e:  # noqa: BLE001 - a requestor that vanished mid-transfer
            logger.debug(f"Could not answer a selection request: {e}")
            return

        if logger.isEnabledFor(logging.DEBUG):
            # get_atom_name is a round trip to the server, so it is only worth
            # paying for when the log will actually show it.
            logger.debug(
                f"selection request target={self._display.get_atom_name(target)} "
                f"requestor=0x{ev.requestor.id:x} served={served} content={content}"
            )
        if content:
            self._record_read(ev.requestor.id)

    def _record_read(self, requestor_id: int) -> None:
        with self._cond:
            self._reads.append((self._read_seq, requestor_id))
            self._read_seq += 1
            if len(self._reads) > 512:
                del self._reads[:256]
            self._cond.notify_all()
