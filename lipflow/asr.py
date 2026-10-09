"""Russian dictation: a ready-made offline recogniser, with no training of our own.

Lipflow's camera path is English (the LRS3 lip reader). There is no public *visual* Russian
model, and the closest available checkpoint does not actually read lips: on the MuAViC Russian
example, decoding video-only exactly the way MuAViC's own `MODALITIES=video` does it (the audio
features zeroed) gives incoherent Russian - 7 of the 28 words of the reference reading - while
the same model with the real audio track transcribes it correctly. So Russian is recognised
from the microphone, through GigaAM v3 (MIT), and the camera plays no part.

onnx-asr (MIT) runs GigaAM's ONNX export on the CPU with numpy + onnxruntime alone: no
PyTorch, no FFmpeg, no fine-tuning. Install it with `uv sync --extra ru`.

The `e2e` GigaAM variants punctuate and capitalise on their own, so Russian needs no LLM pass.
"""
from __future__ import annotations

import numpy as np

MODEL = "gigaam-v3-e2e-ctc"
RATE = 16_000
INSTALL_HINT = "Russian needs the optional recogniser (GigaAM v3 via onnx-asr): uv sync --extra ru"


def available() -> bool:
    """Is the optional recogniser installed? Cheap: answers while the menu is built."""
    try:
        import onnx_asr  # noqa: F401
    except ImportError:
        return False
    return True


class RussianASR:
    """GigaAM v3, used exactly as published. Loads in a few seconds, then ~0.7 s per sentence."""

    def __init__(self, model: str = MODEL):
        if not available():
            raise RuntimeError(INSTALL_HINT)
        import onnx_asr
        self.model_name = model
        self.model = onnx_asr.load_model(model)

    def describe(self) -> str:
        return f"{self.model_name} (Russian, on the microphone)"

    def read(self, wave: np.ndarray) -> str:
        """16 kHz mono float32 -> Russian text, already punctuated and capitalised."""
        w = np.asarray(wave, dtype=np.float32).reshape(-1)
        if w.size == 0:
            return ""
        return (self.model.recognize(w, sample_rate=RATE) or "").strip()

    def warmup(self):
        self.read(np.zeros(RATE // 2, dtype=np.float32))
