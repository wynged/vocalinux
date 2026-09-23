"""Tests for the whisper.cpp keep-warm heartbeat that holds the model in VRAM."""

import sys
import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    saved = dict(sys.modules)
    yield
    added = set(sys.modules.keys()) - set(saved.keys())
    for k in added:
        del sys.modules[k]
    for k, v in saved.items():
        if k not in sys.modules or sys.modules[k] is not v:
            sys.modules[k] = v


for _k in ["vosk", "whisper", "torch", "pyaudio", "pywhispercpp", "pywhispercpp.model"]:
    if _k not in sys.modules:
        sys.modules[_k] = MagicMock()
if "gi" not in sys.modules:
    sys.modules["gi"] = MagicMock()
if "gi.repository" not in sys.modules:
    sys.modules["gi.repository"] = MagicMock()

from vocalinux.common_types import RecognitionState  # noqa: E402
from vocalinux.speech_recognition.recognition_manager import (  # noqa: E402
    SpeechRecognitionManager,
)


def _make_manager(interval=0.05):
    manager = SpeechRecognitionManager.__new__(SpeechRecognitionManager)
    manager.engine = "whisper_cpp"
    manager.state = RecognitionState.IDLE
    manager.model = MagicMock()
    manager._model_lock = threading.Lock()
    manager.whispercpp_keep_warm_seconds = interval
    manager._last_model_use = time.monotonic() - interval
    manager._keep_warm_thread = None
    return manager


def _run_briefly(manager, seconds=0.3):
    manager._start_keep_warm()
    time.sleep(seconds)
    manager.whispercpp_keep_warm_seconds = 0
    manager._keep_warm_thread.join(timeout=2)
    assert not manager._keep_warm_thread.is_alive()


def test_beats_when_idle_and_passes_no_decode_params():
    manager = _make_manager()
    _run_briefly(manager)
    assert manager.model.transcribe.call_count >= 2
    for call in manager.model.transcribe.call_args_list:
        # pywhispercpp keeps any override for later calls, so the beat must pass none
        assert call.kwargs == {}
        assert len(call.args) == 1


def test_does_not_beat_while_recognition_is_active():
    manager = _make_manager()
    manager.state = RecognitionState.LISTENING
    _run_briefly(manager)
    manager.model.transcribe.assert_not_called()


def test_does_not_wait_on_a_held_model_lock():
    manager = _make_manager()
    manager._model_lock.acquire()
    try:
        _run_briefly(manager)
    finally:
        manager._model_lock.release()
    manager.model.transcribe.assert_not_called()


def test_does_not_beat_before_the_model_has_been_idle_long_enough():
    manager = _make_manager(interval=10)
    manager._last_model_use = time.monotonic()
    manager._start_keep_warm()
    time.sleep(0.2)
    manager.model.transcribe.assert_not_called()


def test_zero_interval_disables_the_heartbeat():
    manager = _make_manager(interval=0)
    manager._start_keep_warm()
    assert manager._keep_warm_thread is None


def test_a_failed_beat_does_not_kill_the_heartbeat():
    manager = _make_manager()
    manager.model.transcribe.side_effect = RuntimeError("boom")
    _run_briefly(manager)
    assert manager.model.transcribe.call_count >= 2
    assert not manager._model_lock.locked()
