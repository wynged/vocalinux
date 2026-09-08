"""
Tests for hybrid shortcut mode.

Hybrid mode is push-to-talk and toggle at once on a single key: hold it to
speak, or double-tap it to latch hands-free dictation on until the next
double-tap. These tests cover the two halves separately --

- the keyboard backend, which decides whether a press is the start of a hold
  or the close of a double-tap, and
- the tray indicator, which owns the latch and the deferred stop that keeps a
  double-tap from being chopped into two useless recordings.
"""

import sys
import time
import unittest
from unittest.mock import MagicMock, patch

from vocalinux.ui.keyboard_backends.base import SHORTCUT_MODES
from vocalinux.ui.keyboard_backends.pynput_backend import PynputKeyboardBackend


def _hybrid_backend():
    """A pynput backend in hybrid mode whose key matching always fires."""
    backend = PynputKeyboardBackend(shortcut="left_alt+left_alt", mode="hybrid")
    key = MagicMock(name="alt_l")
    backend._matches_configured_modifier = lambda k: True
    backend._normalize_modifier_key = lambda k: key
    # Run callbacks inline instead of in a thread, so assertions do not race.
    return backend, key


def _record(backend):
    """Register recording callbacks and return the list of event names."""
    events = []
    backend.register_toggle_callback(lambda: events.append("toggle"))
    backend.register_press_callback(lambda: events.append("press"))
    backend.register_release_callback(lambda: events.append("release"))
    return events


def _settle():
    """Let the backend's callback threads run."""
    time.sleep(0.15)


class TestHybridModeRegistered(unittest.TestCase):
    def test_hybrid_is_a_supported_mode(self):
        self.assertIn("hybrid", SHORTCUT_MODES)

    def test_backend_accepts_hybrid(self):
        backend = PynputKeyboardBackend(mode="hybrid")
        self.assertEqual(backend.mode, "hybrid")


class TestHybridBackendRouting(unittest.TestCase):
    """A press is the start of a hold or the close of a double-tap, never both."""

    def test_hold_fires_press_then_release(self):
        backend, key = _hybrid_backend()
        events = _record(backend)

        backend._on_press(key)
        time.sleep(0.05)
        backend._on_release(key)
        _settle()

        self.assertEqual(events, ["press", "release"])

    def test_two_quick_taps_fire_a_single_toggle(self):
        backend, key = _hybrid_backend()
        events = _record(backend)

        backend._on_press(key)
        backend._on_release(key)
        backend._on_press(key)  # within the double-tap window
        backend._on_release(key)
        _settle()

        # First tap starts a hold, second closes the double-tap. The second
        # press must NOT also fire a press callback.
        self.assertEqual(events.count("toggle"), 1)
        self.assertEqual(events.count("press"), 1)

    def test_slow_second_tap_is_not_a_double_tap(self):
        backend, key = _hybrid_backend()
        backend.double_tap_threshold = 0.05
        events = _record(backend)

        backend._on_press(key)
        backend._on_release(key)
        time.sleep(0.1)  # longer than the window
        backend._on_press(key)
        backend._on_release(key)
        _settle()

        self.assertEqual(events.count("toggle"), 0)
        self.assertEqual(events.count("press"), 2)

    def test_auto_repeat_during_a_hold_is_not_a_double_tap(self):
        backend, key = _hybrid_backend()
        events = _record(backend)

        backend._on_press(key)
        for _ in range(5):  # X11 auto-repeat, no release in between
            time.sleep(0.02)
            backend._on_press(key)
        backend._on_release(key)
        _settle()

        self.assertEqual(events.count("toggle"), 0)

    def test_release_is_delivered_in_hybrid_mode(self):
        backend, key = _hybrid_backend()
        events = _record(backend)

        backend._on_press(key)
        backend._on_release(key)
        _settle()

        self.assertIn("release", events)


# --- Tray indicator: the latch and the deferred stop -----------------------

_GI_MODULES = ("gi", "gi.repository")


def _drop_tray_indicator_imports():
    """Forget any cached tray_indicator, so it re-binds to the current gi."""
    for mod in [k for k in list(sys.modules) if "tray_indicator" in k]:
        del sys.modules[mod]


class TestHybridLatch(unittest.TestCase):
    """The tray indicator's hybrid handlers."""

    def setUp(self):
        # Mock GTK/GI so this runs without a display server, and put sys.modules
        # back in tearDown -- leaving a mocked gi behind breaks whichever test
        # file pytest happens to run next.
        self._saved_gi = {name: sys.modules.get(name) for name in _GI_MODULES}
        mock_gi = MagicMock()
        mock_gi_repository = MagicMock()
        mock_gi_repository.GLib.idle_add.side_effect = (
            lambda func, *args: func(*args) or False
        )
        sys.modules["gi"] = mock_gi
        sys.modules["gi.repository"] = mock_gi_repository
        _drop_tray_indicator_imports()

        from vocalinux.common_types import RecognitionState

        self.RecognitionState = RecognitionState

        self.patchers = [
            patch("os.makedirs"),
            patch("vocalinux.ui.tray_indicator.ConfigManager"),
            patch("vocalinux.ui.tray_indicator.KeyboardShortcutManager"),
            patch("vocalinux.ui.tray_indicator.SuspendHandler"),
        ]
        (
            self.mock_makedirs,
            self.mock_config_class,
            self.mock_ksm_class,
            self.mock_suspend,
        ) = [p.start() for p in self.patchers]

        self.mock_config = MagicMock()
        self.mock_config.get_str.side_effect = lambda section, key, default=None: (
            "hybrid" if key == "mode" else "left_alt+left_alt"
        )
        self.mock_config_class.return_value = self.mock_config

        # The engine is a state machine here: start/stop move it, as the real
        # one does, so the handlers are exercised against believable states.
        self.engine = MagicMock()
        self.engine.state = RecognitionState.IDLE

        def start(mode="toggle"):
            self.engine.state = RecognitionState.LISTENING
            self.engine.started_mode = mode

        def stop():
            self.engine.state = RecognitionState.IDLE

        self.engine.start_recognition.side_effect = start
        self.engine.stop_recognition.side_effect = stop

        from vocalinux.ui.tray_indicator import TrayIndicator

        self.tray = TrayIndicator(speech_engine=self.engine, text_injector=MagicMock())
        # A short window keeps the deferred-stop tests quick.
        self.tray.shortcut_manager.backend_instance.double_tap_threshold = 0.05

    def tearDown(self):
        self.tray._cancel_hybrid_stop()
        for p in self.patchers:
            p.stop()

        _drop_tray_indicator_imports()
        for name, module in self._saved_gi.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_hybrid_mode_registers_all_three_callbacks(self):
        ksm = self.tray.shortcut_manager
        ksm.register_toggle_callback.assert_called_with(self.tray._toggle_hands_free)
        ksm.register_press_callback.assert_called_with(self.tray._hybrid_press)
        ksm.register_release_callback.assert_called_with(self.tray._hybrid_release)

    def test_unknown_mode_falls_back_to_push_to_talk(self):
        """A mode this build does not know must not leave the key dead."""
        self.mock_config.get_str.side_effect = lambda section, key, default=None: (
            "some_future_mode" if key == "mode" else "left_alt+left_alt"
        )
        ksm = self.tray.shortcut_manager
        ksm.reset_mock()

        self.tray._setup_keyboard_shortcuts()

        ksm.register_press_callback.assert_called_with(self.tray._start_recognition)
        ksm.register_release_callback.assert_called_with(self.tray._stop_recognition)

    def test_hold_records_then_stops_on_release(self):
        self.tray._hybrid_press()
        self.assertEqual(self.engine.state, self.RecognitionState.LISTENING)
        self.assertEqual(self.engine.started_mode, "push_to_talk")

        time.sleep(0.1)  # held longer than the double-tap window
        self.tray._hybrid_release()
        self.assertEqual(self.engine.state, self.RecognitionState.IDLE)

    def test_short_tap_defers_the_stop_but_still_stops(self):
        self.tray._hybrid_press()
        self.tray._hybrid_release()  # released immediately

        # Still recording: the stop is waiting to see if a second tap arrives.
        self.assertEqual(self.engine.state, self.RecognitionState.LISTENING)

        time.sleep(0.3)
        self.assertEqual(self.engine.state, self.RecognitionState.IDLE)

    def test_double_tap_latches_hands_free_and_keeps_the_audio(self):
        self.tray._hybrid_press()
        self.tray._hybrid_release()
        self.tray._toggle_hands_free()  # the second tap

        self.assertTrue(self.tray._hybrid_latched)
        # The session the first tap started is converted, not restarted.
        self.engine.set_recognition_mode.assert_called_once_with("toggle")
        self.engine.start_recognition.assert_called_once()

        # The deferred stop from the first tap must not fire.
        time.sleep(0.3)
        self.assertEqual(self.engine.state, self.RecognitionState.LISTENING)

    def test_double_tap_from_idle_starts_hands_free(self):
        self.tray._toggle_hands_free()

        self.assertTrue(self.tray._hybrid_latched)
        self.engine.start_recognition.assert_called_once_with(mode="toggle")

    def test_second_double_tap_stops_and_unlatches(self):
        self.tray._toggle_hands_free()
        self.tray._toggle_hands_free()

        self.assertFalse(self.tray._hybrid_latched)
        self.assertEqual(self.engine.state, self.RecognitionState.IDLE)

    def test_holding_the_key_while_latched_does_nothing(self):
        self.tray._toggle_hands_free()
        self.engine.start_recognition.reset_mock()

        self.tray._hybrid_press()
        self.tray._hybrid_release()
        time.sleep(0.3)

        self.engine.start_recognition.assert_not_called()
        self.engine.stop_recognition.assert_not_called()
        self.assertEqual(self.engine.state, self.RecognitionState.LISTENING)

    def test_recognition_ending_elsewhere_clears_the_latch(self):
        self.tray._toggle_hands_free()
        self.assertTrue(self.tray._hybrid_latched)

        # e.g. the suspend handler stopping recognition behind our back
        self.tray._on_recognition_state_changed(self.RecognitionState.IDLE)

        self.assertFalse(self.tray._hybrid_latched)

    def test_latch_survives_the_listening_processing_cycle(self):
        self.tray._toggle_hands_free()

        self.tray._on_recognition_state_changed(self.RecognitionState.PROCESSING)
        self.tray._on_recognition_state_changed(self.RecognitionState.LISTENING)

        self.assertTrue(self.tray._hybrid_latched)


if __name__ == "__main__":
    unittest.main()
