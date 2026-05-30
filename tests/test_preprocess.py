import os
import numpy as np
import subprocess
import tempfile
import pytest
import sbobinator


FFMPEG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist", "ffmpeg.exe"))


def _make_noisy_wav(path, sr=16000, duration_s=3):
    """Genera sine + rumore bianco con ffmpeg."""
    subprocess.run([
        FFMPEG, "-f", "lavfi", "-i",
        f"sine=frequency=440:duration={duration_s}",
        "-f", "lavfi", "-i",
        f"anoisesrc=color=white:duration={duration_s}:amplitude=0.1",
        "-filter_complex", "[0:0][1:0]amix=inputs=2:duration=longest",
        "-ac", "1", "-ar", str(sr), "-y", path
    ], check=True, capture_output=True)


def test_preprocess_audio_no_flags_returns_path():
    """Nessun flag attivo -> ritorna il path originale, nessun file extra."""
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "in.wav")
        _make_noisy_wav(wav)
        out_path, sr = sbobinator.preprocess_audio(wav)
        assert out_path == wav
        assert sr == 16000
        # nessun file .cleaned.wav creato
        assert not os.path.exists(wav.replace(".wav", ".cleaned.wav"))


@pytest.mark.slow
def test_preprocess_audio_denoise_writes_cleaned_file():
    """Con denoise=True scrive un .cleaned.wav valido."""
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "in.wav")
        _make_noisy_wav(wav, duration_s=2)
        out_path, sr = sbobinator.preprocess_audio(wav, denoise=True)
        assert out_path != wav
        assert os.path.isfile(out_path)
        # il file deve essere caricabile e non vuoto
        cleaned = sbobinator.load_audio_array(out_path)
        assert len(cleaned) > 1000
        assert not np.isnan(cleaned).any()


@pytest.mark.slow
def test_preprocess_audio_separate_runs():
    """Con separate=True (su input mono single-speaker) non deve crashare."""
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "in.wav")
        _make_noisy_wav(wav, duration_s=2)
        out_path, sr = sbobinator.preprocess_audio(wav, separate=True)
        assert os.path.isfile(out_path)
        cleaned = sbobinator.load_audio_array(out_path)
        assert not np.isnan(cleaned).any()
