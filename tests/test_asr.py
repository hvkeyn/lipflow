"""Russian runs on a ready-made recogniser (GigaAM via onnx-asr), never on our own training.

The dependency is optional, so the app must degrade to a clear message rather than a crash.
"""
import pytest

from lipflow import asr, lang


def test_recogniser_is_optional(monkeypatch):
    monkeypatch.setattr(asr, "available", lambda: False)
    with pytest.raises(RuntimeError, match="extra ru"):
        asr.RussianASR()


def test_available_is_a_bool():
    assert isinstance(asr.available(), bool)


def test_the_model_is_a_published_russian_one():
    assert asr.MODEL.startswith("gigaam-v3")
    assert asr.RATE == 16_000
    assert "onnx" in asr.INSTALL_HINT


def test_empty_audio_reads_nothing(monkeypatch):
    """No microphone data must not reach the model at all."""
    import numpy as np

    class Fake:
        model_name = "fake"

        def recognize(self, *a, **k):  # pragma: no cover - must not be called
            raise AssertionError("recognize() should not run on empty audio")

    r = asr.RussianASR.__new__(asr.RussianASR)
    r.model = Fake()
    r.model_name = "fake"
    assert r.read(np.zeros(0, dtype=np.float32)) == ""


def test_the_tray_menu_offers_both_languages():
    assert lang.available() == ["en", "ru"]
    assert lang.profile("ru").name == "Русский"


def test_the_windows_app_still_imports():
    """The Russian branch is wired through win/app.py; a syntax slip there breaks the tray."""
    import lipflow.win.app  # noqa: F401


def _bare_app(hud_shown, asr_impl):
    """A Lipflow without tk, camera or threads: ui() runs inline so the HUD calls are visible."""
    from lipflow.win.app import Lipflow

    app = Lipflow.__new__(Lipflow)
    app.ru_asr = asr_impl
    app.hud = type("HUD", (), {"show": staticmethod(lambda *a, **k: hud_shown.append(a))})()
    app.ui = lambda fn, *a, **k: fn(*a, **k)
    return app


def _recording(level, seconds=2.0):
    import numpy as np

    from lipflow.camera import Recording

    rate = 16_000
    rec = Recording(started=0.0)  # ts stays empty: the camera produced no frames at all
    rec.audio = [(100.0 + i / rate, np.full(1600, level, np.float32))
                 for i in range(0, int(rate * seconds), 1600)]
    return rec


def test_russian_does_not_need_the_camera():
    """The span comes from the microphone's own clock, so a camera that sends no frames
    (switched off, blocked, or absent) must not break Russian dictation."""

    class FakeASR:
        seen = None

        def read(self, wave):
            FakeASR.seen = wave
            return "привет"

    app = _bare_app([], FakeASR())
    assert app._ru_candidates(_recording(0.05)) == ["привет"]
    assert FakeASR.seen is not None and len(FakeASR.seen) >= 16_000  # about two seconds of audio


def test_a_silent_microphone_is_reported_as_such():
    """A muted input still opens a stream and still hands back dither (~3e-5). That is not a
    recognition failure, and the user must be told which of the two it is."""
    shown = []

    class NeverCalled:
        def read(self, wave):  # pragma: no cover - the level check must stop us first
            raise AssertionError("the recogniser should not run on a silent capture")

    app = _bare_app(shown, NeverCalled())
    assert app._ru_candidates(_recording(3e-5)) is None
    assert shown and shown[0][0] == "error" and shown[0][1] == lang.t("mic_silent_title")


def test_russian_without_audio_reads_nothing():
    shown = []
    app = _bare_app(shown, object())  # never touched
    assert app._ru_candidates(_recording(0.05, seconds=0)) is None
    assert shown and shown[0][1] == lang.t("mic_silent_title")


def test_silence_and_peak():
    import numpy as np

    assert asr.peak(None) == 0.0
    assert asr.peak(np.zeros(0, np.float32)) == 0.0
    assert asr.silence(np.full(1000, 3e-5, np.float32))  # dither: a muted microphone
    assert not asr.silence(np.full(1000, 0.05, np.float32))
