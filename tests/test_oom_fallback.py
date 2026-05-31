"""Regressione: ripiego automatico GPU -> CPU quando la GPU va in out-of-memory.

Verifica il caso tipico delle schede con poca VRAM (es. T1000 4 GB): il primo
tentativo su GPU fallisce per OOM, l'app ricarica il modello su CPU, invoca il
callback di ripiego e conclude senza duplicare l'output parziale.
"""
import os
import sys
import tempfile

import sbobinator


def test_gpu_oom_falls_back_to_cpu(monkeypatch):
    import whisper

    monkeypatch.setattr(sbobinator, "check_ffmpeg", lambda: True)
    monkeypatch.setattr(sbobinator, "get_audio_duration", lambda p: 1.0)

    calls = {"n": 0}

    def fake_transcribe(model, audio_path, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            sys.stdout.write("[00:00 --> 00:01] parziale\n")  # output parziale...
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "definitivo"}],
                "language": "it"}

    loaded = {"device": None}

    class FakeModel:
        pass

    def fake_load_model(name, download_root=None, device=None):
        loaded["device"] = device
        return FakeModel()

    monkeypatch.setattr(whisper, "transcribe", fake_transcribe)
    monkeypatch.setattr(whisper, "load_model", fake_load_model)

    done, fb = {}, {}
    tmp = tempfile.mktemp(suffix=".wav")
    cfg = {"device": "cuda", "fp16": True, "threads": None, "mode": "memoria ridotta",
           "reason": "t", "vram_gb": 4.0, "gpu_name": "T1000", "ram_gb": 8,
           "cores": 4, "override": "auto"}

    sbobinator.transcribe(
        tmp,
        (lambda m: None, lambda m, p: None, lambda l: None,
         lambda ok, res, info: done.update(ok=ok)),
        preloaded_model=FakeModel(),
        runtime_config=cfg,
        on_fallback=lambda model, c: fb.update(device=c["device"]),
    )

    txt = os.path.splitext(tmp)[0] + ".txt"
    content = open(txt, encoding="utf-8").read() if os.path.exists(txt) else ""
    try:
        os.remove(txt)
    except OSError:
        pass

    assert calls["n"] == 2                # GPU fallita + retry su CPU
    assert loaded["device"] == "cpu"      # modello ricaricato su CPU
    assert fb.get("device") == "cpu"      # callback di ripiego invocato
    assert done.get("ok") is True         # trascrizione conclusa con successo
    assert "parziale" not in content      # nessun output parziale duplicato
