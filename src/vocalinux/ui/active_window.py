"""
Which window has focus, asked of X directly.

A one-thought dictation (a single tap in hybrid mode) ends when focus leaves
the window it was started in, so the tray polls this a few times a second for
as long as one runs. ``xdotool getactivewindow`` would answer too, but as a
fork per poll; this is one property read on a connection of its own (python-
xlib connections are not thread-safe, so it shares nothing with the clipboard
owner's).

Anything that goes wrong -- no X server, no python-xlib, a window manager that
does not publish ``_NET_ACTIVE_WINDOW`` -- makes ``get()`` answer None, which
the caller reads as "cannot tell" and never as "focus moved".
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ActiveWindow:
    """Reads ``_NET_ACTIVE_WINDOW`` off the root window."""

    def __init__(self):
        self._display = None
        try:
            from Xlib import display

            self._display = display.Display()
            self._root = self._display.screen().root
            self._atom = self._display.intern_atom("_NET_ACTIVE_WINDOW")
        except Exception as e:  # noqa: BLE001 - no X is an answer, not an error
            logger.debug(f"Cannot watch the active window: {e}")
            self.close()

    def get(self) -> Optional[int]:
        """The focused window's id, 0 for none (an empty workspace), None if unknown."""
        if self._display is None:
            return None
        try:
            from Xlib import X

            prop = self._root.get_full_property(self._atom, X.AnyPropertyType)
            if prop is None or len(prop.value) == 0:
                return None
            return int(prop.value[0])
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Reading the active window failed: {e}")
            return None

    def close(self) -> None:
        display, self._display = self._display, None
        if display is not None:
            try:
                display.close()
            except Exception:  # noqa: BLE001
                pass
