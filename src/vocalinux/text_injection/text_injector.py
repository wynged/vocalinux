"""
Text injection module for Vocalinux.

This module is responsible for injecting recognized text into the active
application, supporting both X11 and Wayland environments.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from enum import Enum
from typing import Optional  # noqa: F401

from .clipboard_owner import ClipboardOwner, client_of
from .ibus_engine import (
    IBusTextInjector,
    is_ibus_active_input_method,
    is_ibus_available,
    is_ibus_daemon_running,
)

logger = logging.getLogger(__name__)


def _leave_i3_binding_mode(environment=None) -> None:
    """Drop i3 back to its default binding mode before sending keystrokes.

    While i3 sits in a binding mode such as Regolith's "Resize Mode", it grabs
    the whole keyboard, so injected keys fire that mode's bindings instead of
    reaching the focused window -- dictating in resize mode resizes windows at
    random. This must run per injection rather than once per dictation session:
    in toggle mode a session stays open indefinitely, so the mode is usually
    entered long after recognition started.

    The check is name-agnostic (any non-default mode is left) because the mode's
    name is config-defined -- Regolith calls it "Resize Mode", not "resize". A
    no-op when i3 is already in default, and when i3 isn't the window manager.
    """
    # i3 is an X11 window manager. Under Wayland there is no binding mode to
    # leave, and spending a subprocess per injection to discover that is waste.
    if environment is not None and environment not in (
        DesktopEnvironment.X11,
        DesktopEnvironment.X11_IBUS,
    ):
        return

    try:
        state = subprocess.run(
            ["i3-msg", "-t", "get_binding_state"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if state.returncode != 0:
            logger.debug(f"i3-msg get_binding_state failed: {state.stderr.strip()}")
            return

        # Guard the type as well as the parse. text=True makes stdout a str in
        # production, but this runs on the injection path for EVERY segment, and
        # anything unexpected here must degrade to "don't touch the mode" rather
        # than raise into inject_text() and kill dictation outright.
        if not isinstance(state.stdout, str):
            return

        mode = json.loads(state.stdout).get("name", "default")
        if mode == "default":
            return

        logger.info(f"Leaving i3 binding mode '{mode}' before injecting")
        subprocess.run(
            ["i3-msg", "mode", "default"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    # Deliberately broad, for the same reason as the blocked-apps lookup: this is
    # a best-effort courtesy that runs before every injection. Failing to leave a
    # binding mode should cost you a garbled resize, not the whole dictation.
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not leave i3 binding mode: {e}")


class DesktopEnvironment(Enum):
    """Enum representing the desktop environment."""

    X11 = "x11"
    X11_IBUS = "x11-ibus"  # X11 with IBus engine (preferred for non-US layouts)
    WAYLAND = "wayland"
    WAYLAND_XDOTOOL = "wayland-xdotool"  # Wayland with XWayland fallback
    WAYLAND_IBUS = "wayland-ibus"  # Wayland with IBus engine (preferred)
    UNKNOWN = "unknown"


class TextInjector:
    """
    Class for injecting text into the active application.

    This class handles the injection of text into the currently focused
    application window, supporting both X11 and Wayland environments.
    """

    # Set when the window a dictation was meant for has lost focus: what is
    # still being transcribed is parked on the clipboard instead of pasted into
    # whatever has focus now. See divert_to_clipboard. A class default, so it
    # is never missing from the hot path of inject_text.
    _diverted = False

    def __init__(self, wayland_mode: bool = False):
        """
        Initialize the text injector.

        Args:
            wayland_mode: Force Wayland compatibility mode
        """
        self._ibus_injector: Optional[IBusTextInjector] = None
        self.environment = self._detect_environment()
        self._session_environment = self.environment
        self._ibus_ready = False
        self._ibus_init_failed = False
        self._ibus_init_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._clipboard_tool_health = {}
        self._clipboard_timeout = 0.35
        # Clipboard-paste injection (X11): what the clipboard held before we
        # borrowed it, and which injection is entitled to put it back. See
        # _inject_with_xdotool_paste.
        self._paste_lock = threading.Lock()
        self._paste_generation = 0
        self._paste_pending_restore = None  # (generation, previous_text_or_None)
        self._paste_restore_delay = 0.5
        # The clipboard owner, which is what makes a paste verifiable at all;
        # created on first use because it opens its own X connection.
        self._clipboard_owner: Optional[ClipboardOwner] = None
        # A paste that nothing read: its text stays on the clipboard for the
        # user to place by hand, and the next unread segment is appended to it
        # rather than replacing it. See _handle_paste_that_did_not_land.
        self._unlanded = None
        self._last_no_target_notify_ts = 0.0
        # Throttle for "injection suppressed" notifications: a single dictation
        # session injects several segments, and we don't want one notification
        # per segment when the focused app is on the blocklist.
        self._last_block_notify_ts = 0.0

        # Force Wayland mode if requested
        if wayland_mode and self.environment == DesktopEnvironment.X11:
            logger.info("Forcing Wayland compatibility mode")
            self.environment = DesktopEnvironment.WAYLAND
            self._session_environment = self.environment

        logger.info(f"Using text injection for {self.environment.value} environment")

        # Check for required tools
        self._check_dependencies()

        # Test if wtype actually works in this environment
        if (
            self.environment == DesktopEnvironment.WAYLAND
            and hasattr(self, "wayland_tool")
            and self.wayland_tool == "wtype"
        ):
            try:
                # Try a test with wtype
                result = subprocess.run(
                    ["wtype", "test"], stderr=subprocess.PIPE, text=True, check=False
                )
                error_output = result.stderr.lower()
                if "compositor does not support" in error_output or result.returncode != 0:
                    logger.warning(
                        "Wayland compositor does not support virtual "
                        f"keyboard protocol: {error_output}"
                    )
                    if shutil.which("xdotool"):
                        logger.info("Automatically switching to XWayland fallback with xdotool")
                        self.environment = DesktopEnvironment.WAYLAND_XDOTOOL
                    else:
                        logger.error("No fallback text injection method available")
            except Exception as e:
                logger.warning(f"Error testing wtype: {e}, will try to use it anyway")

        # Verify XWayland fallback works - perform a test injection
        if self.environment == DesktopEnvironment.WAYLAND_XDOTOOL:
            logger.info("Testing XWayland text injection fallback")
            try:
                # Wait a moment to ensure any error messages are displayed before test
                time.sleep(0.5)
                # Try xdotool in more verbose mode for better diagnostics
                self._test_xdotool_fallback()
            except Exception as e:
                logger.error(f"XWayland fallback test failed: {e}")

    def stop(self) -> None:
        """
        Clean up resources and restore previous state.

        Call this when shutting down Vocalinux.
        """
        with self._state_lock:
            if self._ibus_injector:
                logger.info("Stopping IBus text injector")
                self._ibus_injector.stop()
                self._ibus_injector = None
            self._ibus_ready = False
            if self._clipboard_owner is not None:
                # An X selection dies with the process that owns it, and what
                # we are still holding at shutdown is dictation the user has
                # not placed yet. Hand it to xclip, whose forked owner outlives
                # us, rather than taking it to the grave.
                held = self._clipboard_owner.current_text()
                if held:
                    self._set_x11_clipboard(held, os.environ.copy())
                self._clipboard_owner.stop()
                self._clipboard_owner = None

    def _detect_environment(self) -> DesktopEnvironment:
        """
        Detect the current desktop environment (X11 or Wayland).

        Returns:
            The detected desktop environment
        """
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "wayland":
            return DesktopEnvironment.WAYLAND
        elif session_type == "x11":
            return DesktopEnvironment.X11
        else:
            # Try to detect based on other methods
            if "WAYLAND_DISPLAY" in os.environ:
                return DesktopEnvironment.WAYLAND
            elif "DISPLAY" in os.environ:
                return DesktopEnvironment.X11
            else:
                logger.warning("Could not detect desktop environment, defaulting to X11")
                return DesktopEnvironment.X11

    def _check_dependencies(self):
        """Check for the required tools for text injection."""
        ibus_requested = False

        # Prefer IBus on both X11 and Wayland - it sends Unicode directly,
        # bypassing keyboard layout issues entirely
        if is_ibus_available():
            # Check if IBus is the active input method (not just installed)
            # This is important because IBus may be installed but not being used,
            # e.g., when the user has configured ydotool or Fcitx instead
            if not is_ibus_active_input_method():
                logger.info(
                    "IBus is installed but not the active input method. "
                    "Falling back to alternative text injection method."
                )
            # Check if ibus-daemon is running before attempting setup
            elif not is_ibus_daemon_running():
                logger.info(
                    "IBus daemon not running. This is normal on some desktop environments "
                    "(e.g., KDE Plasma). Using alternative text injection method. "
                    "For IBus setup, see: https://github.com/jatinkrmalik/vocalinux/wiki/IBus-Setup"
                )
            else:
                try:
                    self._ibus_injector = IBusTextInjector(auto_activate=False)
                    ibus_requested = True
                except Exception as e:
                    logger.warning(f"IBus initialization failed: {e}, trying alternatives")
        if self.environment == DesktopEnvironment.X11:
            # Check for xdotool
            if not shutil.which("xdotool"):
                if ibus_requested:
                    self._start_ibus_initialization()
                    return
                logger.error("xdotool not found. Please install it with: sudo apt install xdotool")
                raise RuntimeError("Missing required dependency: xdotool")
        else:
            # Fallback: Check for wtype or ydotool for Wayland
            wtype_available = shutil.which("wtype") is not None
            ydotool_available = shutil.which("ydotool") is not None
            xdotool_available = shutil.which("xdotool") is not None

            if ydotool_available:
                # Verify ydotoold daemon is running before selecting ydotool
                try:
                    subprocess.run(
                        ["ydotool", "type", ""],
                        check=True,
                        stderr=subprocess.PIPE,
                        timeout=2,
                    )
                    self.wayland_tool = "ydotool"
                    logger.info(f"Using {self.wayland_tool} for Wayland text injection")
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                    FileNotFoundError,
                ):
                    if wtype_available:
                        self.wayland_tool = "wtype"
                        logger.info(
                            "Using "
                            f"{self.wayland_tool} for Wayland text injection "
                            "(ydotoold not running)"
                        )
                    else:
                        logger.warning("ydotool found but ydotoold daemon not running")
            elif wtype_available:
                self.wayland_tool = "wtype"
                logger.info(f"Using {self.wayland_tool} for Wayland text injection")
            elif xdotool_available:
                # Fallback to xdotool with XWayland
                self.environment = DesktopEnvironment.WAYLAND_XDOTOOL
                logger.info(
                    "No native Wayland tools found. Using xdotool with XWayland as fallback"
                )
            else:
                if ibus_requested:
                    self._start_ibus_initialization()
                    return
                logger.error(
                    "No text injection tools found. Please install one of:\n"
                    "- IBus (recommended, usually pre-installed)\n"
                    "- wtype: sudo apt install wtype (GNOME/Sway)\n"
                    "- ydotool: sudo apt install ydotool (works on all Wayland compositors)\n"
                    "- xdotool: sudo apt install xdotool (X11/XWayland only)\n"
                    "\n"
                    "For KDE Plasma Wayland users: wtype is not supported. "
                    "Install ydotool or wl-copy for clipboard fallback:\n"
                    "  sudo apt install ydotool\n"
                    "  sudo systemctl enable --now ydotoold\n"
                    "Or for clipboard fallback: sudo apt install wl-copy"
                )
                raise RuntimeError("Missing required dependencies for text injection")

        if ibus_requested:
            self._start_ibus_initialization()

    def _start_ibus_initialization(self) -> None:
        if self._ibus_injector is None or self._ibus_init_thread is not None:
            return

        if self.environment == DesktopEnvironment.WAYLAND and not hasattr(self, "wayland_tool"):
            if shutil.which("wtype"):
                self.wayland_tool = "wtype"
            elif shutil.which("ydotool"):
                self.wayland_tool = "ydotool"

        self._ibus_init_failed = False
        self._ibus_init_thread = threading.Thread(
            target=self._initialize_ibus_in_background,
            daemon=True,
        )
        self._ibus_init_thread.start()
        logger.info("Starting IBus warmup in background")

    def _initialize_ibus_in_background(self) -> None:
        if self._ibus_injector is None:
            return

        try:
            # Keep the Vocalinux IBus process warm without selecting it as the
            # user's active keyboard engine. Activation is scoped to each text
            # commit so normal typing keeps layout-specific compose/dead keys.
            self._ibus_injector.prepare_engine()
            with self._state_lock:
                self._ibus_ready = True
                self._ibus_init_failed = False
                if self._session_environment == DesktopEnvironment.X11:
                    self.environment = DesktopEnvironment.X11_IBUS
                else:
                    self.environment = DesktopEnvironment.WAYLAND_IBUS
            logger.info(
                f"Using IBus for {self.environment.value} text injection (best compatibility)"
            )
        except Exception as e:
            with self._state_lock:
                self._ibus_ready = False
                self._ibus_init_failed = True
            logger.warning(f"IBus initialization failed: {e}, continuing with fallback")

    def _get_clipboard_tools(self):
        tools = []
        if self._session_environment == DesktopEnvironment.WAYLAND and shutil.which("wl-copy"):
            tools.append("wl-copy")
        if shutil.which("xclip"):
            tools.append("xclip")
        if shutil.which("xsel"):
            tools.append("xsel")
        if self._session_environment != DesktopEnvironment.WAYLAND and shutil.which("wl-copy"):
            tools.append("wl-copy")
        return tools

    def _run_clipboard_command(self, tool: str, text: str) -> bool:
        if tool == "wl-copy":
            subprocess.run(
                ["wl-copy", text],
                check=True,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._clipboard_timeout,
            )
            return True

        if tool == "xclip":
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text,
                check=True,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._clipboard_timeout,
            )
            return True

        if tool == "xsel":
            subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=text,
                check=True,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._clipboard_timeout,
            )
            return True

        return False

    def _switch_to_non_ibus_backend(self) -> bool:
        """Switch from IBus mode to a non-IBus backend for runtime fallback."""
        if self.environment == DesktopEnvironment.X11_IBUS:
            if shutil.which("xdotool"):
                with self._state_lock:
                    self.environment = DesktopEnvironment.X11
                logger.warning("IBus injection failed, switching to X11 xdotool fallback")
                return True

            logger.error("IBus fallback failed: xdotool is not available on X11")
            return False

        if self.environment == DesktopEnvironment.WAYLAND_IBUS:
            ydotool_available = shutil.which("ydotool") is not None
            wtype_available = shutil.which("wtype") is not None
            xdotool_available = shutil.which("xdotool") is not None

            if ydotool_available:
                try:
                    subprocess.run(
                        ["ydotool", "type", ""],
                        check=True,
                        stderr=subprocess.PIPE,
                        timeout=2,
                    )
                    with self._state_lock:
                        self.wayland_tool = "ydotool"
                        self.environment = DesktopEnvironment.WAYLAND
                    logger.warning("IBus injection failed, switching to Wayland ydotool fallback")
                    return True
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                    FileNotFoundError,
                ):
                    logger.debug("ydotool fallback unavailable (daemon not running)")

            if wtype_available:
                with self._state_lock:
                    self.wayland_tool = "wtype"
                    self.environment = DesktopEnvironment.WAYLAND
                logger.warning("IBus injection failed, switching to Wayland wtype fallback")
                return True

            if xdotool_available:
                with self._state_lock:
                    self.environment = DesktopEnvironment.WAYLAND_XDOTOOL
                logger.warning("IBus injection failed, switching to XWayland xdotool fallback")
                return True

            logger.error(
                "IBus fallback failed: no Wayland text injection tools available "
                "(ydotool, wtype, xdotool)"
            )
            return False

        return True

    def _test_xdotool_fallback(self):
        """Test if xdotool is working correctly with XWayland."""
        try:
            # Get the DISPLAY environment variable for XWayland
            xwayland_display = os.environ.get("DISPLAY", ":0")
            logger.debug(f"Using DISPLAY={xwayland_display} for XWayland")

            # Try using xdotool with explicit DISPLAY setting
            test_env = os.environ.copy()
            test_env["DISPLAY"] = xwayland_display

            # Check if we can get active window (less intrusive test)
            window_id = subprocess.run(
                ["xdotool", "getwindowfocus"],
                env=test_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if window_id.returncode != 0 or "failed" in window_id.stderr.lower():
                logger.warning(f"XWayland detection test failed: {window_id.stderr}")
                # Try to force XWayland environment more explicitly
                test_env["GDK_BACKEND"] = "x11"
            else:
                logger.debug("XWayland test successful")
        except Exception as e:
            logger.error(f"Failed to test XWayland fallback: {e}")

    def _try_recover_from_fallback(self):
        """
        Try to recover from xdotool fallback mode by re-checking for better tools.

        This allows switching to ydotool if the daemon was started after initial detection,
        or to wtype if the compositor now supports virtual keyboard.

        Returns:
            True if a better tool was found and environment was updated, False otherwise
        """
        if self.environment != DesktopEnvironment.WAYLAND_XDOTOOL:
            return False

        logger.info("Checking for better Wayland text injection tools...")

        # Check for ydotool with daemon running
        if shutil.which("ydotool"):
            try:
                subprocess.run(
                    ["ydotool", "type", ""],
                    check=True,
                    stderr=subprocess.PIPE,
                    timeout=2,
                )
                self.wayland_tool = "ydotool"
                self.environment = DesktopEnvironment.WAYLAND
                logger.info("Recovered to ydotool - ydotoold daemon is now running")
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                logger.debug("ydotool available but daemon not running")

        # Check for wtype with compositor support
        if shutil.which("wtype"):
            try:
                result = subprocess.run(
                    ["wtype", "test"],
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                error_output = result.stderr.lower()
                if "compositor does not support" not in error_output and result.returncode == 0:
                    self.wayland_tool = "wtype"
                    self.environment = DesktopEnvironment.WAYLAND
                    logger.info("Recovered to wtype - compositor now supports virtual keyboard")
                    return True
            except Exception as e:
                logger.debug(f"Error testing wtype: {e}")

        logger.debug("No better tools available, continuing with xdotool fallback")
        return False

    def _copy_to_clipboard(self, text: str) -> bool:
        """
        Copy text to clipboard.

        This is useful for:
        - Fallback when injection fails on unsupported compositors (like KDE Plasma)
        - Always-on clipboard copy so users can paste recognized text elsewhere

        Args:
            text: The text to copy to clipboard

        Returns:
            True if clipboard copy was successful, False otherwise
        """
        logger.info("Copying text to clipboard")

        for tool in self._get_clipboard_tools():
            if self._clipboard_tool_health.get(tool) is False:
                continue

            try:
                if self._run_clipboard_command(tool, text):
                    self._clipboard_tool_health[tool] = True
                    logger.info(f"Text copied to clipboard using {tool}")
                    return True
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                FileNotFoundError,
            ) as e:
                self._clipboard_tool_health[tool] = False
                logger.warning(f"{tool} failed: {e}")

        logger.warning(
            "Clipboard copy failed. Install wl-copy (Wayland) or xclip/xsel "
            "to enable clipboard functionality."
        )
        return False

    def _should_copy_to_clipboard(self) -> bool:
        """Check if copy-to-clipboard setting is enabled."""
        try:
            import json

            config_path = os.path.expanduser("~/.config/vocalinux/config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                return config.get("text_injection", {}).get("copy_to_clipboard", False)
        except Exception as e:
            logger.debug(f"Could not read copy_to_clipboard setting: {e}")
        return False

    def _show_clipboard_fallback_notification(self):
        """Show a desktop notification when text is copied to clipboard as fallback."""
        try:
            subprocess.Popen(
                [
                    "notify-send",
                    "-i",
                    "edit-paste",
                    "-a",
                    "Vocalinux",
                    "Text copied to clipboard",
                    "Text injection failed - paste with Ctrl+V",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.debug(f"Could not show clipboard notification: {e}")

    def _get_blocked_apps(self) -> list:
        """Read the per-app injection blocklist from config.

        Each entry is matched case-insensitively as a substring against both the
        focused window's class and its title, so a single entry like "gather"
        catches the Gather desktop app (window class) *and* a browser tab titled
        "Gather" (window title) without muting the rest of the browser.
        """
        try:
            import json

            config_path = os.path.expanduser("~/.config/vocalinux/config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                entries = config.get("text_injection", {}).get("blocked_apps", [])
                return [str(e).lower() for e in entries if str(e).strip()]
        except Exception as e:
            logger.debug(f"Could not read blocked_apps setting: {e}")
        return []

    def _get_focused_window_identity(self):
        """Return ``(class, title)`` of the focused X11 window, both lowercased.

        Returns ``("", "")`` when the information is unavailable (pure Wayland,
        or any xdotool failure). Used to decide whether injection is blocked for
        the currently focused application.
        """
        return self._probe_focused_window()[1:]

    def _probe_focused_window(self):
        """Return ``(window_id, class, title)`` for the focused X11 window.

        The id is what says which X client a later clipboard read came from,
        so the paste path takes all three from one probe instead of asking
        twice. ``(None, "", "")`` when the window cannot be identified.
        """
        if self.environment not in (
            DesktopEnvironment.X11,
            DesktopEnvironment.X11_IBUS,
            DesktopEnvironment.WAYLAND_XDOTOOL,
        ):
            return None, "", ""

        env = os.environ.copy()
        if self.environment == DesktopEnvironment.WAYLAND_XDOTOOL:
            env["GDK_BACKEND"] = "x11"
            env["QT_QPA_PLATFORM"] = "xcb"
            if not env.get("DISPLAY"):
                env["DISPLAY"] = ":0"

        def _run(args):
            try:
                result = subprocess.run(
                    args,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    timeout=2,
                )
                stdout = result.stdout
                return stdout.strip() if isinstance(stdout, str) else ""
            # Deliberately broad. This is an ADVISORY gate that runs before every
            # injection: if identifying the focused window fails for any reason,
            # the correct outcome is "not blocked" (inject normally), never an
            # exception escaping into inject_text() and killing dictation over a
            # window-title lookup.
            except Exception:  # noqa: BLE001
                return ""

        window_id = _run(["xdotool", "getactivewindow"])
        if not window_id:
            return None, "", ""
        # `xdotool getwindowclassname` only exists from xdotool 3.2021 on;
        # Ubuntu's 3.20160805 answers "Unknown command", which used to make
        # the class silently empty here. WM_CLASS via xprop works everywhere
        # and carries both halves (instance, class) -- e.g. Chrome's app
        # windows are ("crx_<id>", "Google-chrome"), so both are returned,
        # space-joined, for substring matching.
        window_class = " ".join(
            re.findall(r'"([^"]*)"', _run(["xprop", "-id", window_id, "WM_CLASS"]))
        )
        window_title = _run(["xdotool", "getwindowname", window_id])
        try:
            numeric_id = int(window_id)
        except ValueError:
            numeric_id = None
        return numeric_id, window_class.lower(), window_title.lower()

    def _is_injection_blocked_for_focused_window(self) -> bool:
        """Return True if the focused app is on the user's injection blocklist.

        When blocked, dictated text must NOT be typed. A throttled desktop
        notification is shown instead so the suppression is visible without
        spamming one notification per dictated segment.
        """
        blocked = self._get_blocked_apps()
        if not blocked:
            return False

        window_class, window_title = self._get_focused_window_identity()
        haystack = f"{window_class} {window_title}"
        match = next((pattern for pattern in blocked if pattern in haystack), None)
        if match is None:
            return False

        logger.info(
            "Injection suppressed: focused window matches blocklist entry "
            f"'{match}' (class='{window_class}', title='{window_title}')"
        )

        now = time.monotonic()
        if now - self._last_block_notify_ts > 4.0:
            self._notify_injection_blocked(window_title or window_class or "focused window")
            self._last_block_notify_ts = now
        return True

    def _notify_injection_blocked(self, app_label: str):
        """Show a desktop notification that injection was suppressed for an app."""
        try:
            subprocess.Popen(
                [
                    "notify-send",
                    "-i",
                    "microphone-sensitivity-muted",
                    "-a",
                    "Vocalinux",
                    "Dictation suppressed",
                    f"'{app_label}' is on the injection blocklist — text was not typed.",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.debug(f"Could not show injection-blocked notification: {e}")

    def inject_text(self, text: str) -> bool:
        """
        Inject text into the currently focused application.

        Args:
            text: The text to inject

        Returns:
            True if injection was successful, False otherwise
        """
        if not text or not text.strip():
            logger.debug("Empty text provided, skipping injection")
            return True

        if self._diverted:
            self._park_on_clipboard(text)
            return True

        # Per-app blocklist: in apps like Gather, single keystrokes are commands
        # (movement, interactions), so typing a dictated sentence fires a flurry
        # of shortcuts. When the focused window matches the user's blocklist,
        # suppress injection entirely (notify-only) instead of typing.
        if self._is_injection_blocked_for_focused_window():
            return True

        # Leave any i3 binding mode (e.g. resize) first, so the keystrokes below
        # reach the focused window instead of driving the mode's bindings.
        _leave_i3_binding_mode(self.environment)

        logger.info(f"Starting text injection: '{text}' (length: {len(text)})")
        logger.debug(f"Environment: {self.environment}")

        # Get information about the current window/application
        self._log_current_window_info()

        # Note: No shell escaping needed - subprocess is called with list arguments,
        # which passes text directly without shell interpretation
        logger.debug(f"Text to inject: '{text}'")

        # Re-check for available tools in Wayland fallback mode
        # This allows switching to ydotool if the daemon was started after initial detection
        if self.environment == DesktopEnvironment.WAYLAND_XDOTOOL:
            self._try_recover_from_fallback()

        try:
            with self._state_lock:
                current_env = self.environment
                ibus_injector = self._ibus_injector

            if (
                current_env == DesktopEnvironment.WAYLAND_IBUS
                or current_env == DesktopEnvironment.X11_IBUS
            ):
                if ibus_injector is not None:
                    result = ibus_injector.inject_text(text)
                    if result:
                        logger.info("Text injection completed successfully")
                        if self._should_copy_to_clipboard():
                            threading.Thread(
                                target=self._copy_to_clipboard,
                                args=(text,),
                                daemon=True,
                            ).start()
                        return True

                    logger.warning(
                        "IBus runtime injection failed. Falling back to non-IBus backend."
                    )
                    if not self._switch_to_non_ibus_backend():
                        raise RuntimeError(
                            "IBus injection failed and no non-IBus fallback is available"
                        )
                else:
                    logger.error("IBus injector not initialized, trying non-IBus fallback")
                    if not self._switch_to_non_ibus_backend():
                        raise RuntimeError(
                            "IBus injector not initialized and no non-IBus fallback is available"
                        )

            with self._state_lock:
                current_env = self.environment

            if (
                current_env == DesktopEnvironment.X11
                or current_env == DesktopEnvironment.WAYLAND_XDOTOOL
            ):
                self._inject_with_xdotool(text)
            else:
                try:
                    self._inject_with_wayland_tool(text)
                except subprocess.CalledProcessError as e:
                    stderr_msg = e.stderr.strip() if e.stderr else "No stderr output"
                    logger.warning(
                        f"Wayland tool failed: {e}. stderr: {stderr_msg}. Falling back to xdotool"
                    )
                    if "compositor does not support" in str(
                        e
                    ).lower() + " " + stderr_msg.lower() and shutil.which("xdotool"):
                        logger.info(
                            "Switching to XWayland fallback - will re-check for better tools"
                        )
                        with self._state_lock:
                            self.environment = DesktopEnvironment.WAYLAND_XDOTOOL
                        self._inject_with_xdotool(text)
                    else:
                        raise
            logger.info("Text injection completed successfully")

            if self._should_copy_to_clipboard():
                threading.Thread(
                    target=self._copy_to_clipboard,
                    args=(text,),
                    daemon=True,
                ).start()

            return True
        except Exception as e:
            logger.error(f"Failed to inject text: {e}", exc_info=True)

            try:
                if self._copy_to_clipboard(text):
                    logger.info("Text copied to clipboard as fallback - user can paste manually")
                    self._show_clipboard_fallback_notification()
                    return True
            except Exception as clipboard_error:
                logger.debug(f"Clipboard fallback also failed: {clipboard_error}")

            try:
                from ..ui.audio_feedback import play_error_sound

                play_error_sound()
            except ImportError:
                logger.warning("Could not import audio feedback module")
            return False

    # How long to wait after the paste chord for something to read the
    # clipboard. Measured on this desktop: a GTK entry, Chrome's omnibox and a
    # WezTerm pane each asked within ~30-50 ms, so this is generous, and it is
    # only ever waited out when the paste found nowhere to go.
    PASTE_READ_TIMEOUT = 0.4
    # How long dictation left on the clipboard keeps collecting later segments
    # before the next one counts as a fresh thought. One utterance arrives as
    # several segments a second or two apart, so this only has to outlast a
    # pause; longer would risk appending to words the user has already placed.
    UNLANDED_TTL = 20.0

    # Window classes whose paste chord is Ctrl+Shift+V, because Ctrl+V is a
    # control byte to the program inside. Matched as substrings of WM_CLASS.
    TERMINAL_WINDOW_CLASSES = (
        "wezterm",
        "xterm",
        "rxvt",
        "alacritty",
        "kitty",
        "terminal",
        "terminator",
        "tilix",
        "konsole",
        "st-256color",
        "foot",
    )

    def _paste_injection_enabled(self) -> bool:
        """``text_injection.paste_injection`` from config; defaults to on."""
        try:
            import json

            config_path = os.path.expanduser("~/.config/vocalinux/config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                return bool(config.get("text_injection", {}).get("paste_injection", True))
        except Exception as e:
            logger.debug(f"Could not read paste_injection setting: {e}")
        return True

    def _paste_verify_enabled(self) -> bool:
        """``text_injection.paste_verify`` from config; defaults to on.

        Turning it off gives up knowing whether a paste landed and puts the
        clipboard back in xclip's hands.
        """
        try:
            config_path = os.path.expanduser("~/.config/vocalinux/config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                return bool(config.get("text_injection", {}).get("paste_verify", True))
        except Exception as e:  # noqa: BLE001 - an unreadable config must not stop injection
            logger.debug(f"Could not read paste_verify setting: {e}")
        return True

    def _paste_chord_for_focused_window(self) -> str:
        return self._paste_chord_for_window_class(self._get_focused_window_identity()[0])

    def _paste_chord_for_window_class(self, window_class: str) -> str:
        if any(marker in window_class for marker in self.TERMINAL_WINDOW_CLASSES):
            return "ctrl+shift+v"
        return "ctrl+v"

    def _read_x11_clipboard(self, env) -> Optional[str]:
        """The clipboard's text, or None if it holds none (empty, or an image)."""
        try:
            result = subprocess.run(
                ["xclip", "-o", "-selection", "clipboard"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1,
            )
        except Exception as e:  # noqa: BLE001 - advisory; losing the old clipboard is not fatal
            logger.debug(f"Could not read clipboard: {e}")
            return None
        if result.returncode != 0 or not isinstance(result.stdout, str):
            return None
        return result.stdout

    def _set_x11_clipboard(self, text: str, env) -> bool:
        # xclip forks a child that owns the selection until someone else claims
        # it, and that child inherits stderr. With stderr=PIPE, run() waits
        # for EOF that never comes and the 0.35 s clipboard timeout fires --
        # which is why _copy_to_clipboard marks xclip unhealthy on this box.
        # Both streams go to DEVNULL here so the parent's exit is the end.
        try:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
                timeout=1,
            )
            return True
        except Exception as e:  # noqa: BLE001 - any failure means "type it instead"
            logger.warning(f"Could not set clipboard for paste injection: {e}")
            return False

    def _inject_with_xdotool_paste(self, text: str, env) -> bool:
        """Put ``text`` on the clipboard and press the focused window's paste chord.

        Returns False, with nothing typed, when paste injection is disabled or
        the clipboard could not be set -- the caller then types the text
        instead. The clipboard's previous text is put back a moment after the
        paste (an image or other non-text content cannot be saved with xclip
        and is simply left replaced).

        Segments can arrive faster than the restore delay, so the restore is
        owned by a generation counter: a newer injection reuses the *original*
        saved text rather than snapshotting the previous segment, and an older
        restore that fires after a newer injection does nothing.

        When Vocalinux owns the clipboard itself, it can also tell whether the
        paste landed anywhere -- see _handle_paste_that_did_not_land.
        """
        if not self._paste_injection_enabled():
            return False

        owner = self._selection_owner()
        if owner is None and not shutil.which("xclip"):
            logger.debug("xclip not available; typing instead of pasting")
            return False

        window_id, window_class, _title = self._probe_focused_window()

        with self._paste_lock:
            self._paste_generation += 1
            generation = self._paste_generation
            unlanded = self._live_unlanded()
            if self._paste_pending_restore is not None:
                previous = self._paste_pending_restore[1]
            elif unlanded is not None:
                # Still holding text the user has not placed yet: what to put
                # back is what THEY had, not the dictation sitting there now.
                previous = unlanded[1]
            else:
                previous = self._read_x11_clipboard(env)
            self._paste_pending_restore = (generation, previous)

        owner_holds = owner is not None and owner.set_text(text)
        if not owner_holds and not self._set_x11_clipboard(text, env):
            with self._paste_lock:
                if self._paste_pending_restore and self._paste_pending_restore[0] == generation:
                    self._paste_pending_restore = None
            return False

        if not owner_holds:
            owner = None
        chord = self._paste_chord_for_window_class(window_class)
        landed = self._press_paste(chord, env, owner, window_id)
        if landed is None:
            self._schedule_clipboard_restore(generation, env)
            return False

        if not landed:
            self._handle_paste_that_did_not_land(text, generation, previous, owner)
            return True

        with self._paste_lock:
            self._unlanded = None
        logger.info(
            f"Text injected via clipboard paste ({chord}): "
            f"'{text[:20]}...' ({len(text)} chars)"
        )
        self._schedule_clipboard_restore(generation, env)
        return True

    def _press_paste(self, chord: str, env, owner, window_id):
        """Send the paste chord.

        Returns True if the text landed somewhere, False if nothing read the
        clipboard, and None if the keystroke itself failed (the caller then
        types instead). Without an owner there is nothing to observe, so the
        answer is True: the paste is assumed to have worked, exactly as it was
        before any of this could be checked.
        """

        def press():
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", chord],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )

        try:
            if owner is None:
                press()
                return True
            with owner.watch() as watch:
                press()
                return watch.landed(client_of(window_id), self.PASTE_READ_TIMEOUT)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Paste keystroke failed ({e}); falling back to typing")
            return None

    def _live_unlanded(self):
        """The unplaced dictation still on the clipboard, or None.

        It stops being live once something else claims the clipboard, or after
        long enough that the next dictation is plainly a new thought rather
        than the rest of the last one. Whether the *user* has since pasted it
        by hand is not knowable: their Ctrl+V is a clipboard read like any
        other, and a clipboard manager makes one of those every second.
        """
        unlanded = self._unlanded
        if unlanded is None:
            return None
        _text, _previous, when, owner = unlanded
        if time.monotonic() - when > self.UNLANDED_TTL:
            return None
        if not owner.owns():
            return None
        return unlanded

    def _handle_paste_that_did_not_land(
        self, text: str, generation: int, previous, owner, notice=None
    ):
        """Nothing read the clipboard, so the text went nowhere. Leave it there.

        A dictation arrives as several segments whenever the speaker pauses,
        and each one pastes on its own. Replacing the clipboard per segment
        would leave only the last sentence to recover, so segments that find
        nowhere to land accumulate instead.
        """
        with self._paste_lock:
            if self._paste_pending_restore and self._paste_pending_restore[0] == generation:
                # Do not put the old clipboard back: the dictation on it is the
                # only copy the user has, and pasting it by hand is the point.
                self._paste_pending_restore = None
            unlanded = self._live_unlanded()

        combined = f"{unlanded[0]} {text}" if unlanded is not None else text
        if unlanded is not None and not owner.set_text(combined):
            combined = text
        logger.warning(
            f"Nothing read the clipboard after the paste: {len(combined)} chars are "
            "waiting there for the user to place"
        )
        with self._paste_lock:
            self._unlanded = (combined, previous, time.monotonic(), owner)

        now = time.monotonic()
        if now - self._last_no_target_notify_ts > 4.0:
            self._notify_paste_did_not_land(*(notice or ()))
            self._last_no_target_notify_ts = now

    def divert_to_clipboard(self) -> None:
        """Park every segment on the clipboard until end_divert() is called.

        For a dictation whose window has lost focus: the speaker has moved on,
        but the last thing they said is still being transcribed, and pasting it
        into the window they moved to is exactly wrong. Dropping it would be
        wrong too, so it waits on the clipboard for them to place by hand.
        """
        self._diverted = True

    def end_divert(self) -> None:
        """Go back to pasting into the focused window."""
        self._diverted = False

    def _park_on_clipboard(self, text: str) -> None:
        """Put a segment on the clipboard, after any already parked, and say so."""
        text = text.strip()
        logger.info(f"Window changed mid-dictation; parking {len(text)} chars on the clipboard")
        notice = (
            "Dictation is on the clipboard",
            "You switched windows before it finished \u2014 press Ctrl+V where you want it.",
        )
        owner = self._selection_owner()
        if owner is None:
            if self._copy_to_clipboard(text):
                self._notify_paste_did_not_land(*notice)
            return

        env = os.environ.copy()
        with self._paste_lock:
            self._paste_generation += 1  # an older segment's restore must not undo this
            generation = self._paste_generation
            unlanded = self._live_unlanded()
            if self._paste_pending_restore is not None:
                previous = self._paste_pending_restore[1]
            elif unlanded is not None:
                previous = unlanded[1]
            else:
                previous = self._read_x11_clipboard(env)
            self._paste_pending_restore = None
        # With nothing parked yet the text has to go on the clipboard here; with
        # something parked, the handler below appends to it.
        if unlanded is None and not owner.set_text(text):
            logger.warning("Could not park dictation on the clipboard; it is lost")
            return
        self._handle_paste_that_did_not_land(text, generation, previous, owner, notice)

    def _notify_paste_did_not_land(
        self,
        title: str = "Nowhere to paste",
        body: str = "The text couldn't find a place to land \u2014 press Ctrl+V where you want it.",
    ):
        """Tell the user their words are on the clipboard and where to put them."""
        try:
            subprocess.Popen(
                [
                    "notify-send",
                    "-i",
                    "edit-paste",
                    "-a",
                    "Vocalinux",
                    title,
                    body,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:  # noqa: BLE001 - a missing notify-send must not break dictation
            logger.debug(f"Could not show the nowhere-to-paste notification: {e}")

    def _selection_owner(self):
        """The clipboard owner to paste through, or None to fall back to xclip.

        Owning the selection in-process is the only way to see whether the
        paste was read, so it is also how the text gets onto the clipboard.
        """
        if not self._paste_verify_enabled():
            return None
        with self._state_lock:
            if self._clipboard_owner is None:
                self._clipboard_owner = ClipboardOwner()
            owner = self._clipboard_owner
        return owner if owner.start() else None

    def _schedule_clipboard_restore(self, generation: int, env) -> None:
        def restore():
            time.sleep(self._paste_restore_delay)
            with self._paste_lock:
                pending = self._paste_pending_restore
                if pending is None or pending[0] != generation:
                    return  # a newer injection owns the clipboard now
                self._paste_pending_restore = None
            previous = pending[1]
            if previous is None:
                return
            if self._set_x11_clipboard(previous, env):
                logger.debug("Restored the clipboard after paste injection")

        threading.Thread(target=restore, daemon=True, name="clipboard-restore").start()

    def _inject_with_xdotool(self, text: str):
        """
        Inject text using xdotool for X11 environments.

        Args:
            text: The text to inject
        """
        # Create environment with explicit X11 settings for Wayland compatibility
        env = os.environ.copy()

        if self.environment == DesktopEnvironment.WAYLAND_XDOTOOL:
            # Force X11 backend for XWayland
            env["GDK_BACKEND"] = "x11"
            env["QT_QPA_PLATFORM"] = "xcb"
            # Ensure DISPLAY is set correctly for XWayland
            if "DISPLAY" not in env or not env["DISPLAY"]:
                env["DISPLAY"] = ":0"

            logger.debug(f"Using XWayland with DISPLAY={env['DISPLAY']}")

            # Add a small delay to ensure text is injected properly
            time.sleep(0.3)  # Increased delay for better reliability

            # Try to ensure the window has focus using more robust approach
            try:
                # Get current active window
                active_window = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

                if active_window.returncode == 0 and active_window.stdout.strip():
                    window_id = active_window.stdout.strip()
                    # Focus explicitly on that window
                    subprocess.run(
                        ["xdotool", "windowactivate", "--sync", window_id],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    # Wait a moment for the focus to take effect
                    time.sleep(0.2)
            except Exception as e:
                logger.debug(f"Window focus command failed: {e}")

        # Paste first: one Ctrl+V lands a whole utterance at once, where typing
        # runs at ~12 ms per character (a 1000-char dictation took 11.8 s to
        # type against 3.2 s to transcribe). Typing remains the fallback.
        if self._inject_with_xdotool_paste(text, env):
            return

        # Inject text using xdotool
        try:
            max_retries = 2
            logger.debug(f"Starting xdotool injection with {max_retries} max retries")

            for retry in range(max_retries + 1):
                try:
                    # Inject in smaller chunks to avoid issues with very long text
                    chunk_size = 20  # Reduced chunk size for better reliability
                    total_chunks = (len(text) + chunk_size - 1) // chunk_size
                    logger.debug(
                        f"Splitting text into {total_chunks} chunks of max {chunk_size} chars"
                    )

                    for i in range(0, len(text), chunk_size):
                        chunk = text[i : i + chunk_size]
                        chunk_num = (i // chunk_size) + 1

                        # First try with clearmodifiers
                        cmd = ["xdotool", "type", "--clearmodifiers", chunk]
                        logger.debug(f"Injecting chunk {chunk_num}/{total_chunks}: '{chunk}'")

                        subprocess.run(
                            cmd,
                            env=env,
                            check=True,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=5,
                        )

                        # Add a larger delay between chunks
                        if i + chunk_size < len(text):
                            time.sleep(0.1)

                    logger.info(
                        f"Text injected using xdotool: '{text[:20]}...' ({len(text)} chars)"
                    )
                    break  # Successfully injected
                except subprocess.CalledProcessError as chunk_error:
                    if retry < max_retries:
                        logger.warning(
                            f"Retrying text injection (attempt {retry + 1}/{max_retries}): "
                            f"{chunk_error.stderr}"
                        )
                        time.sleep(0.5)  # Wait before retry
                    else:
                        logger.error(f"Final attempt failed: {chunk_error.stderr}")
                        raise  # Re-raise on final attempt
                except subprocess.TimeoutExpired:
                    if retry < max_retries:
                        logger.warning(
                            f"Text injection timeout, retrying (attempt {retry + 1}/{max_retries})"
                        )
                        time.sleep(0.5)
                    else:
                        logger.error("Text injection timed out on final attempt")
                        raise

            # Release any modifier keys that may be stuck down. Use keyup on the
            # modifier keysyms directly rather than sending Escape: pressing
            # Escape does not actually reset modifiers (--clearmodifiers only
            # temporarily clears them while the keystroke is sent), and a stray
            # Escape is destructive in TUIs that bind it to "cancel input" (e.g.
            # the Claude Code prompt clears the in-progress message on Escape).
            try:
                subprocess.run(
                    [
                        "xdotool",
                        "keyup",
                        "Control_L",
                        "Control_R",
                        "Shift_L",
                        "Shift_R",
                        "Alt_L",
                        "Alt_R",
                        "Super_L",
                        "Super_R",
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass  # Ignore any errors from this command
        except subprocess.CalledProcessError as e:
            logger.error(f"xdotool error: {e.stderr}")
            raise

    def _has_non_ascii(self, text: str) -> bool:
        """Check if text contains any non-ASCII characters."""
        try:
            text.encode("ascii")
            return False
        except UnicodeEncodeError:
            return True

    def _inject_via_clipboard_paste(self, text: str) -> bool:
        """
        Inject text by copying to clipboard and simulating Ctrl+V with ydotool.

        This is the workaround for ydotool's inability to type non-ASCII/Unicode
        characters (accented letters, CJK, etc.) because ydotool simulates evdev
        key events which only cover US ASCII keycodes. See issue #362.

        Note: this temporarily overwrites the user's clipboard. There is no
        attempt to restore it afterward, as there is no safe race-free way to
        do so on Wayland.

        Returns:
            True if successful, False otherwise
        """
        logger.debug(
            "Using clipboard-paste injection for non-ASCII text "
            "(user clipboard will be temporarily overwritten)"
        )

        if not self._copy_to_clipboard(text):
            logger.warning("Could not copy text to clipboard for paste injection")
            return False

        # Simulate Ctrl+V via ydotool using evdev keycodes:
        # KEY_LEFTCTRL=29, KEY_V=47; value 1=press, 0=release.
        # wtype is intentionally not handled here: wtype uses the Wayland
        # virtual-keyboard protocol which supports Unicode natively, so it
        # never needs the clipboard-paste workaround.
        try:
            subprocess.run(
                ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
                check=True,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )
            logger.info(f"Text injected via clipboard paste: '{text[:20]}...' ({len(text)} chars)")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Paste simulation failed: {e}")
            return False

    def _inject_with_wayland_tool(self, text: str):
        """
        Inject text using a Wayland-compatible tool (wtype or ydotool).

        For ydotool: if the text contains non-ASCII characters (accented
        letters like á, é, ú, CJK characters, etc.), uses clipboard-based
        injection instead, because ydotool simulates evdev key events which
        only cover US ASCII keycodes. See issue #362.

        Args:
            text: The text to inject

        Raises:
            subprocess.CalledProcessError: If the tool fails, with stderr captured
        """
        # ydotool can only handle ASCII characters because it works at the
        # evdev keycode level. For non-ASCII text, use clipboard paste instead.
        if self.wayland_tool == "ydotool" and self._has_non_ascii(text):
            logger.info(
                "Text contains non-ASCII characters, using clipboard paste "
                "for ydotool (evdev keycodes are ASCII-only)"
            )
            if self._inject_via_clipboard_paste(text):
                return
            logger.warning(
                "Clipboard paste failed, falling back to ydotool type "
                "(non-ASCII characters may be dropped)"
            )

        if self.wayland_tool == "wtype":
            cmd = ["wtype", text]
        else:  # ydotool
            cmd = ["ydotool", "type", text]

        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            # Re-raise with stderr preserved for better diagnostics
            raise subprocess.CalledProcessError(
                e.returncode, e.cmd, output=e.output, stderr=e.stderr
            ) from e

        logger.info(
            f"Text injected using {self.wayland_tool}: '{text[:20]}...' ({len(text)} chars)"
        )

    def _inject_keyboard_shortcut(self, shortcut: str) -> bool:
        """
        Inject a keyboard shortcut.

        Args:
            shortcut: The keyboard shortcut to inject (e.g., "ctrl+z", "ctrl+a")

        Returns:
            True if injection was successful, False otherwise
        """
        logger.debug(f"Injecting keyboard shortcut: {shortcut}")

        _leave_i3_binding_mode(self.environment)

        try:
            if (
                self.environment == DesktopEnvironment.X11
                or self.environment == DesktopEnvironment.WAYLAND_XDOTOOL
            ):
                return self._inject_shortcut_with_xdotool(shortcut)
            else:
                return self._inject_shortcut_with_wayland_tool(shortcut)
        except Exception as e:
            logger.error(f"Failed to inject keyboard shortcut '{shortcut}': {e}")
            return False

    def _inject_shortcut_with_xdotool(self, shortcut: str) -> bool:
        """
        Inject a keyboard shortcut using xdotool.

        Args:
            shortcut: The keyboard shortcut to inject

        Returns:
            True if successful, False otherwise
        """
        # Create environment with explicit X11 settings for Wayland compatibility
        env = os.environ.copy()

        if self.environment == DesktopEnvironment.WAYLAND_XDOTOOL:
            env["GDK_BACKEND"] = "x11"
            env["QT_QPA_PLATFORM"] = "xcb"
            if "DISPLAY" not in env or not env["DISPLAY"]:
                env["DISPLAY"] = ":0"

        try:
            cmd = ["xdotool", "key", "--clearmodifiers", shortcut]
            subprocess.run(cmd, env=env, check=True, stderr=subprocess.PIPE, text=True)
            logger.debug(f"Keyboard shortcut '{shortcut}' injected successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"xdotool shortcut error: {e.stderr}")
            return False

    def _inject_shortcut_with_wayland_tool(self, shortcut: str) -> bool:
        """
        Inject a keyboard shortcut using a Wayland-compatible tool.

        Args:
            shortcut: The keyboard shortcut to inject

        Returns:
            True if successful, False otherwise
        """
        if self.wayland_tool == "wtype":
            # wtype doesn't support key combinations directly, so we can't implement this easily
            logger.warning("Keyboard shortcuts not supported with wtype")
            return False
        elif self.wayland_tool == "ydotool":
            try:
                cmd = ["ydotool", "key", shortcut]
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
                logger.debug(f"Keyboard shortcut '{shortcut}' injected successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"ydotool shortcut error: {e.stderr}")
                return False
        else:
            logger.warning(f"Keyboard shortcuts not supported with {self.wayland_tool}")
            return False

    def _log_current_window_info(self):
        """Log information about the current window/application for debugging."""
        try:
            if (
                self.environment == DesktopEnvironment.X11
                or self.environment == DesktopEnvironment.WAYLAND_XDOTOOL
            ):
                self._log_x11_window_info()
            else:
                logger.debug("Window info logging not available for pure Wayland")
        except Exception as e:
            logger.debug(f"Could not get window info: {e}")

    def _log_x11_window_info(self):
        """Log X11 window information."""
        env = os.environ.copy()

        if self.environment == DesktopEnvironment.WAYLAND_XDOTOOL:
            env["GDK_BACKEND"] = "x11"
            env["QT_QPA_PLATFORM"] = "xcb"
            if "DISPLAY" not in env or not env["DISPLAY"]:
                env["DISPLAY"] = ":0"

        try:
            # Get active window ID
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=2,
            )
            window_id = result.stdout.strip()
            logger.debug(f"Active window ID: {window_id}")

            # Get window name
            result = subprocess.run(
                ["xdotool", "getwindowname", window_id],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=2,
            )
            window_name = result.stdout.strip()
            logger.info(f"Target window: '{window_name}' (ID: {window_id})")

            # Get window class
            result = subprocess.run(
                ["xdotool", "getwindowclassname", window_id],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=2,
            )
            window_class = result.stdout.strip()
            logger.debug(f"Window class: {window_class}")

            # Get window PID
            result = subprocess.run(
                ["xdotool", "getwindowpid", window_id],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=2,
            )
            window_pid = result.stdout.strip()
            logger.debug(f"Window PID: {window_pid}")

            # Try to get process name
            try:
                with open(f"/proc/{window_pid}/comm", "r") as f:
                    process_name = f.read().strip()
                logger.info(f"Target process: {process_name} (PID: {window_pid})")
            except Exception:
                pass

        except subprocess.TimeoutExpired:
            logger.warning("Timeout getting window information")
        except subprocess.CalledProcessError as e:
            logger.debug(f"xdotool command failed: {e.stderr}")
        except Exception as e:
            logger.debug(f"Error getting window info: {e}")
