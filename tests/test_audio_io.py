import os
import numpy as np
import subprocess
import tempfile
import sbobinator


FFMPEG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist", "ffmpeg.exe"))
print(f"FFMPEG path: {FFMPEG}, exists: {os.path.exists(FFMPEG)}")


def _make_test_wav(path, duration_s=2, freq=440, sr=16000):
    """Genera un sine con ffmpeg per testare."""
    subprocess.run([
        FFMPEG,
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration_s}",
        "-ac", "1", "-ar", str(sr), "-y", path
    ], check=True, capture_output=True)


def test_load_audio_array_shape_and_type():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "test.wav")
        _make_test_wav(wav, duration_s=2, sr=16000)
        audio = sbobinator.load_audio_array(wav)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        # 2 secondi a 16 kHz = ~32000 campioni (±5%)
        assert 30000 < len(audio) < 34000


def test_load_audio_array_normalized():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "test.wav")
        _make_test_wav(wav, duration_s=1, sr=16000)
        audio = sbobinator.load_audio_array(wav)
        # audio deve stare in [-1, 1]
        assert audio.min() > -1.01
        assert audio.max() < 1.01
        # ha ampiezza non trascurabile (sine non silenzioso)
        assert audio.max() > 0.01
