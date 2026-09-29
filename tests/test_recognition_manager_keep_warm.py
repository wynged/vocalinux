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
from vocalinux.speech_recognition import recognition_manager as rm  # noqa: E402
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
    manager._model_on_gpu = True
    manager._whispercpp_model_path = "/models/ggml-medium.bin"
    manager._whispercpp_model_kwargs = {"n_threads": 4}
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


def test_a_cpu_model_gets_no_heartbeat():
    # Nothing to evict from VRAM; the beat would only burn CPU on a laptop battery
    manager = _make_manager()
    manager._model_on_gpu = False
    manager._start_keep_warm()
    assert manager._keep_warm_thread is None


MIB = 1024


def _fdinfo(client_id, vram_kib, gtt_kib, pdev="0000:2d:00.0"):
    return (
        "pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\n"
        f"drm-client-id:\t{client_id}\ndrm-pdev:\t{pdev}\n"
        f"drm-resident-gtt:\t{gtt_kib} KiB\ndrm-resident-vram:\t{vram_kib} KiB\n"
    )


def test_residency_counts_a_client_shared_by_several_fds_once(tmp_path):
    (tmp_path / "0").write_text("pos:\t0\nflags:\t0100000\n")
    (tmp_path / "11").write_text(_fdinfo(439, 864212, 1048192))
    (tmp_path / "12").write_text(_fdinfo(439, 864212, 1048192))
    assert rm._gpu_memory_residency(str(tmp_path)) == ("0000:2d:00.0", 864212, 1048192)


def test_residency_is_none_without_a_vram_gtt_split(tmp_path):
    (tmp_path / "0").write_text("pos:\t0\n")
    (tmp_path / "1").write_text(
        "drm-driver:\ti915\ndrm-client-id:\t3\ndrm-total-system0:\t5 KiB\n"
    )
    assert rm._gpu_memory_residency(str(tmp_path)) is None


def test_free_vram_reads_amdgpu_sysfs(tmp_path):
    dev = tmp_path / "0000:2d:00.0"
    dev.mkdir()
    (dev / "mem_info_vram_total").write_text(str(8 * 1024**3))
    (dev / "mem_info_vram_used").write_text(str(5 * 1024**3))
    assert rm._free_vram_kib("0000:2d:00.0", str(tmp_path)) == 3 * 1024 * MIB
    assert rm._free_vram_kib("0000:99:00.0", str(tmp_path)) is None


def _spilled(monkeypatch, manager, free_kib, after_gtt=2 * MIB):
    old = manager.model
    readings = iter(
        [("0000:2d:00.0", 864 * MIB, 1024 * MIB), ("0000:2d:00.0", 1900 * MIB, after_gtt)]
    )
    monkeypatch.setattr(rm, "_gpu_memory_residency", lambda: next(readings))
    monkeypatch.setattr(rm, "_free_vram_kib", lambda pdev: free_kib)
    new = MagicMock()
    manager._load_model_with_compatible_params = MagicMock(return_value=new)
    return old, new


def test_a_spilled_model_is_reloaded_when_vram_has_room(monkeypatch):
    manager = _make_manager()
    old, new = _spilled(monkeypatch, manager, free_kib=3000 * MIB)
    manager._reload_if_spilled()
    manager._load_model_with_compatible_params.assert_called_once_with(
        "/models/ggml-medium.bin", {"n_threads": 4}
    )
    assert manager.model is new
    assert not manager._model_lock.locked()


def test_a_spilled_model_is_left_alone_while_vram_is_full(monkeypatch):
    # Reloading now would only land the new copy in system RAM as well
    manager = _make_manager()
    old, _ = _spilled(monkeypatch, manager, free_kib=900 * MIB)
    manager._reload_if_spilled()
    manager._load_model_with_compatible_params.assert_not_called()
    assert manager.model is old


def test_no_reload_while_dictating(monkeypatch):
    manager = _make_manager()
    manager.state = RecognitionState.LISTENING
    old, _ = _spilled(monkeypatch, manager, free_kib=3000 * MIB)
    manager._reload_if_spilled()
    manager._load_model_with_compatible_params.assert_not_called()
    assert manager.model is old


def test_a_resident_model_is_not_reloaded(monkeypatch):
    manager = _make_manager()
    resident = ("0000:2d:00.0", 1900 * MIB, 2 * MIB)
    monkeypatch.setattr(rm, "_gpu_memory_residency", lambda: resident)
    monkeypatch.setattr(rm, "_free_vram_kib", lambda pdev: 3000 * MIB)
    manager._load_model_with_compatible_params = MagicMock()
    manager._reload_if_spilled()
    manager._load_model_with_compatible_params.assert_not_called()


def test_a_reload_that_still_spills_backs_off(monkeypatch):
    manager = _make_manager()
    _spilled(monkeypatch, manager, free_kib=3000 * MIB, after_gtt=1024 * MIB)
    manager._reload_if_spilled()
    assert manager._next_vram_reload_at > time.monotonic() + 500
    assert manager._vram_reload_backoff == 2 * SpeechRecognitionManager._RELOAD_BACKOFF_SECONDS
    # Still spilled on the next beat, but inside the backoff: no second reload
    _spilled(monkeypatch, manager, free_kib=3000 * MIB)
    manager._reload_if_spilled()
    manager._load_model_with_compatible_params.assert_not_called()


def test_a_failed_reload_releases_the_lock(monkeypatch):
    manager = _make_manager()
    _spilled(monkeypatch, manager, free_kib=3000 * MIB)
    manager._load_model_with_compatible_params.side_effect = RuntimeError("vk oom")
    monkeypatch.setattr(rm, "_show_notification", MagicMock())
    manager._reload_if_spilled()
    assert manager.model is None
    assert not manager._model_lock.locked()
    rm._show_notification.assert_called_once()
