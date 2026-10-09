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


def test_russian_does_not_need_the_camera():
    """The span comes from the microphone's own clock, so a camera that sends no frames
    (switched off, blocked, or absent) must not break Russian dictation."""
    import numpy as np

    from lipflow.camera import Recording
    from lipflow.win.app import Lipflow

    class FakeASR:
        seen = None

        def read(self, wave):
            FakeASR.seen = wave
            return "привет"

    app = Lipflow.__new__(Lipflow)  # no tk, no camera, no threads
    app.ru_asr = FakeASR()
    rec = Recording(started=0.0)  # ts stays empty: the camera produced no frames at all
    rate = 16_000
    rec.audio = [(100.0 + i / rate, np.full(1600, 0.01, np.float32)) for i in range(0, rate * 2, 1600)]

    assert app._ru_candidates(rec) == ["привет"]
    assert FakeASR.seen is not None and len(FakeASR.seen) >= rate  # about two seconds of audio


def test_russian_without_audio_reads_nothing():
    from lipflow.camera import Recording
    from lipflow.win.app import Lipflow

    app = Lipflow.__new__(Lipflow)
    app.ru_asr = object()  # never touched
    assert app._ru_candidates(Recording(started=0.0)) is None
