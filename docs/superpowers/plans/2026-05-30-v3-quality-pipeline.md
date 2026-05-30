# Sbobinator v3 — Quality Pipeline (GPU + Denoising) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trasformare Sbobinator in uno strumento adatto a audio forensi rumorosi: accelerazione GPU (CUDA) per velocità, e pipeline di pulizia audio (denoise, separa voci sovrapposte, super‑risoluzione) per qualità su intercettazioni sporche.

**Architecture:** Due fasi sequenziali ma indipendentemente shippabili. Fase A: auto‑detect CUDA, sposta whisper+pyannote su GPU se presente (5–10× speedup, zero nuove dipendenze pesanti). Fase B: pipeline di pre‑processing audio prima di whisper, con DeepFilterNet 3 (denoising) e ClearerVoice‑Studio (MossFormer2 SE/SS + super‑risoluzione). Carico/scarico modelli in sequenza per stare nei 4 GB di VRAM del T1000. UI: due profili (Standard, Intercettazione, Max Qualità) + Avanzato.

**Tech Stack:** Python 3.12, PyTorch 2.x con CUDA 12.1 (wheel `+cu121`), openai-whisper (Large-v3 Turbo), pyannote.audio 4.x (community-1), deepfilternet, clearvoice, ffmpeg, tkinter, PyInstaller, numpy, scipy.

**Branch:** `v3-quality-pipeline` (creato da main aggiornato post-v2).

**Hardware target:** NVIDIA T1000 4 GB VRAM (Turing, sm_75) + 32 GB RAM + AMD Ryzen 3 PRO. Fallback CPU per macchine senza GPU CUDA.

---

## Strategia di test e debug (riferimento per tutte le task)

### Test pyramid
1. **Unit tests pure** (pytest senza modelli): logica deterministica — funzioni di routing device, parsing risultati, formattazione. Veloci (< 1s ciascuno).
2. **Integration tests con modelli reali** (pytest con marker `@pytest.mark.slow`): caricano i modelli su CPU, eseguono su clip sintetiche (sine + rumore), verificano shape/dtype/non-NaN. Lenti (10–60s ciascuno).
3. **End-to-end manuali** (script in `tests/manual/`): script che girano la pipeline completa su audio reali (intercettazione fornita) e producono trascrizione + log per ispezione umana.
4. **Frozen exe verification**: `archive_viewer` per confermare bundling; launch test per startup; ispezione manuale post-build.

### Strategia di debug
- **Logging strutturato**: ogni stage della pipeline scrive su stderr (e su `pipeline.log` in dist quando frozen) con timestamp + nome stage + durata.
- **Salvataggio intermedi opzionale**: quando l'opzione "debug audio" è attiva, salva i WAV intermedi (`<audio>.denoised.wav`, `<audio>.separated_1.wav`, ecc.) per ispezione manuale.
- **Modalità diagnostica frozen**: `Sbobinator.exe --selftest <audio>` esegue la pipeline e scrive il log + i file intermedi accanto all'audio, exit code != 0 se errori. Permette di validare l'exe sul PC target offline.
- **Failure modes documentati**: ogni task elenca i casi di errore conosciuti e come riconoscerli.

### Audio di test
- **Sintetico** (per unit/integration): sine 440Hz @ 16kHz, 5s, mixato con rumore bianco (-10 dB). Generato via ffmpeg in fixture pytest.
- **Reale clean**: `Il pranzo calabrese.mp3` (monologo, dialetto, 1 voce, rumore basso).
- **Reale forense**: intercettazione fornita da Paolo (multi-voce, radio/traffico, dialetto) — quando arriva su Desktop.

### Comandi di verifica chiave
- `pytest tests/ -v -m "not slow"` → unit tests rapidi
- `pytest tests/ -v -m slow` → integration con modelli (richiede modelli scaricati)
- `python -m PyInstaller.utils.cliutils.archive_viewer -l dist/Sbobinator.exe` → ispezione bundling

---

## File structure

### Files modificati
- `sbobinator.py` — main module. Aggiunge: `get_device()`, `denoise_df3()`, `enhance_clearvoice()`, `separate_clearvoice()`, `super_resolve()`, `preprocess_audio()`, modifica `transcribe()` e `diarize()` per device + preprocess. UI: nuovi controlli profilo. CLI: `--selftest`.
- `sbobinator.spec` — collect_all per `deepfilternet`, `clearvoice`, bundle modelli denoiser.
- `requirements.txt` — aggiunge `deepfilternet`, `clearvoice`, `sounddevice`, marker per torch CUDA.
- `.github/workflows/build.yml` — installa torch da index `+cu121`, installa deepfilternet/clearvoice.
- `.gitignore` — aggiunge `df_models/` se non bundle-by-default.

### Files nuovi
- `tests/conftest.py` — fixture pytest (audio sintetico, device mock).
- `tests/test_device.py` — unit test su `get_device()`.
- `tests/test_audio_io.py` — unit test su `load_audio_array()`.
- `tests/test_preprocess.py` — integration test su pipeline pulizia (slow).
- `tests/manual/run_pipeline.py` — script end-to-end su audio reale.

### Files non toccati (ma rilevanti)
- `hf_models/` — modelli pyannote già committati in v2, restano.
- `dist/` `build/` — generati, gitignored.

---

## FASE A — Supporto GPU CUDA

### Task A1: Aggiungere `get_device()` con test

**Files:**
- Modify: `sbobinator.py` (aggiunge helper dopo `get_audio_duration`)
- Create: `tests/test_device.py`
- Create: `tests/conftest.py`

- [ ] **Step A1.1: Setup tests dir + conftest**

Run:
```
New-Item -ItemType Directory -Force -Path tests | Out-Null
```

`tests/conftest.py`:
```python
import pytest

@pytest.fixture
def fake_cuda(monkeypatch):
    """Forza torch.cuda.is_available() a True per i test."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA T1000")
    return True

@pytest.fixture
def fake_no_cuda(monkeypatch):
    """Forza torch.cuda.is_available() a False per i test."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return False
```

- [ ] **Step A1.2: Scrivere i test (devono fallire)**

`tests/test_device.py`:
```python
import sbobinator

def test_get_device_returns_cuda_when_available(fake_cuda):
    assert sbobinator.get_device() == "cuda"

def test_get_device_returns_cpu_when_unavailable(fake_no_cuda):
    assert sbobinator.get_device() == "cpu"

def test_get_device_info_with_cuda(fake_cuda):
    info = sbobinator.get_device_info()
    assert info["device"] == "cuda"
    assert "T1000" in info["name"]

def test_get_device_info_no_cuda(fake_no_cuda):
    info = sbobinator.get_device_info()
    assert info["device"] == "cpu"
    assert info["name"] == "CPU"
```

- [ ] **Step A1.3: Eseguire test (deve fallire con AttributeError)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_device.py -v
```
Expected: FAIL — `module 'sbobinator' has no attribute 'get_device'`

- [ ] **Step A1.4: Implementare `get_device()` e `get_device_info()` in sbobinator.py**

Inserire dopo `get_audio_duration`:
```python
def get_device():
    """Ritorna 'cuda' se disponibile e funzionante, altrimenti 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_device_info():
    """Info per UI: nome del dispositivo + VRAM disponibile (GB)."""
    import torch
    device = get_device()
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        try:
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3), 1)
        except Exception:
            vram_gb = None
        return {"device": "cuda", "name": name, "vram_gb": vram_gb}
    return {"device": "cpu", "name": "CPU", "vram_gb": None}
```

- [ ] **Step A1.5: Eseguire test (deve passare)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_device.py -v
```
Expected: 4 passed

- [ ] **Step A1.6: Commit**

```
git add sbobinator.py tests/test_device.py tests/conftest.py
git commit -m "v3: get_device/get_device_info per auto-detect CUDA + test"
```

**Failure modes:**
- `ImportError torch`: torch non installato. Verifica con `pip list torch`.
- Test crash su `monkeypatch`: pytest non installato. `pip install pytest`.

---

### Task A2: Passare device a Whisper

**Files:**
- Modify: `sbobinator.py` (funzione `transcribe`, chiamata a `whisper.load_model`)

- [ ] **Step A2.1: Trovare la chiamata corrente a whisper.load_model**

Run:
```
.\.venv\Scripts\python.exe -c "import sbobinator,inspect; src=inspect.getsource(sbobinator.transcribe); print([l for l in src.split('\n') if 'load_model' in l])"
```
Expected: stampa una riga simile a `model = whisper.load_model(model_name, download_root=get_models_dir())`

- [ ] **Step A2.2: Modificare la chiamata aggiungendo device**

Trovare in `sbobinator.py` la riga:
```python
        model = whisper.load_model(model_name, download_root=get_models_dir())
```
Sostituire con:
```python
        model = whisper.load_model(model_name, download_root=get_models_dir(), device=get_device())
```

- [ ] **Step A2.3: Test rapido import + costruzione (verifica nessun crash)**

Run:
```
.\.venv\Scripts\python.exe -c "import sbobinator; print('device:', sbobinator.get_device())"
```
Expected: stampa `device: cpu` (nel dev attuale non c'è CUDA) senza errori.

- [ ] **Step A2.4: Commit**

```
git add sbobinator.py
git commit -m "v3: whisper.load_model usa get_device() (CUDA se disponibile)"
```

**Failure modes:**
- `TypeError: load_model() got unexpected keyword argument 'device'`: whisper troppo vecchio. Aggiornare con `pip install -U openai-whisper`.

---

### Task A3: Passare device a pyannote

**Files:**
- Modify: `sbobinator.py` (funzione `diarize`, chiamata a `pipe.to`)

- [ ] **Step A3.1: Localizzare `pipe.to(torch.device("cpu"))` in `diarize`**

Run:
```
.\.venv\Scripts\python.exe -c "import sbobinator,inspect; src=inspect.getsource(sbobinator.diarize); [print(l) for l in src.split('\n') if 'pipe.to' in l or 'torch.device' in l]"
```
Expected: stampa `pipe.to(torch.device(\"cpu\"))`

- [ ] **Step A3.2: Sostituire la riga**

In `sbobinator.py` dentro `diarize`, sostituire:
```python
    pipe.to(torch.device("cpu"))
```
con:
```python
    pipe.to(torch.device(get_device()))
```

- [ ] **Step A3.3: Sintassi OK**

Run:
```
.\.venv\Scripts\python.exe -m py_compile sbobinator.py
```
Expected: nessun output (silenzio = ok).

- [ ] **Step A3.4: Commit**

```
git add sbobinator.py
git commit -m "v3: pyannote usa get_device() (CUDA se disponibile)"
```

**Failure modes:**
- VRAM insufficiente: su GPU < 2 GB whisper+pyannote vanno OOM. Su T1000 (4 GB) il carico sequenziale risolve.

---

### Task A4: UI — indicatore device

**Files:**
- Modify: `sbobinator.py` (App.__init__, aggiunge label sotto al motto)

- [ ] **Step A4.1: Aggiungere la label nell'`__init__`**

Trovare in `sbobinator.py` dentro `App.__init__` la riga:
```python
        tk.Label(frame, text="⚜  Nei secoli fedele  ⚜", font=("Arial", 9, "italic"),
                 fg=CARA_ROSSO, bg=CARA_BLU).pack(pady=(0, 12))
```

Subito DOPO aggiungere:
```python
        info = get_device_info()
        device_text = (
            f"⚡ GPU: {info['name']} ({info['vram_gb']} GB)"
            if info["device"] == "cuda"
            else "💻 CPU (nessuna GPU CUDA rilevata)"
        )
        device_color = CARA_ORO if info["device"] == "cuda" else COL_MUTED
        tk.Label(frame, text=device_text, font=("Arial", 9),
                 fg=device_color, bg=CARA_BLU).pack(pady=(0, 8))
```

- [ ] **Step A4.2: Test headless GUI**

Run:
```
.\.venv\Scripts\python.exe -c "import tkinter,sbobinator; r=tkinter.Tk(); r.withdraw(); sbobinator.App(r); print('GUI OK'); r.destroy()"
```
Expected: `GUI OK`

- [ ] **Step A4.3: Commit**

```
git add sbobinator.py
git commit -m "v3: indicatore GPU/CPU nell'UI"
```

---

### Task A5: Installare torch+cu121 in dev venv

**Files:**
- (Nessun file modificato — setup ambiente)

- [ ] **Step A5.1: Disinstallare torch CPU corrente**

Run:
```
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchaudio
```

- [ ] **Step A5.2: Installare torch+cu121**

Run:
```
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Expected: installazione completata. Download ~2 GB.

- [ ] **Step A5.3: Verificare CUDA disponibile (dev PC potrebbe non avere GPU NVIDIA)**

Run:
```
.\.venv\Scripts\python.exe -c "import torch; print('cuda available:', torch.cuda.is_available()); print('version:', torch.version.cuda)"
```
Expected: `cuda available: False` (dev PC ha AMD, normale) `version: 12.1`

Nota: anche su PC senza GPU NVIDIA, l'installazione torch+cu121 produce un torch CPU funzionante. La differenza si vede solo sui PC con GPU NVIDIA (come quello di Mirko).

- [ ] **Step A5.4: Re-run dei test unit**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_device.py -v
```
Expected: 4 passed (i mock funzionano comunque)

**Failure modes:**
- Errore di rete su `pip install`: il pacchetto torch+cu121 è grande. Riprovare. Se proxy aziendale, configurare prima `pip config set global.proxy ...`.

---

### Task A6: Update requirements.txt + build.yml per CUDA

**Files:**
- Modify: `requirements.txt`
- Modify: `.github/workflows/build.yml`

- [ ] **Step A6.1: requirements.txt**

Sostituire il contenuto di `requirements.txt` con:
```
openai-whisper
tkinter-dnd2
pyannote.audio==4.0.4
torch
torchaudio
--extra-index-url https://download.pytorch.org/whl/cu121
```

- [ ] **Step A6.2: build.yml — install step**

In `.github/workflows/build.yml`, trovare la riga:
```yaml
          pip install openai-whisper pyinstaller tiktoken "pyannote.audio==4.0.4" torchaudio
```

Sostituire con:
```yaml
          pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchaudio
          pip install openai-whisper pyinstaller tiktoken "pyannote.audio==4.0.4"
```

L'ordine è importante: prima torch+cu121, poi le altre dipendenze che NON devono override-are torch.

- [ ] **Step A6.3: Commit**

```
git add requirements.txt .github/workflows/build.yml
git commit -m "v3: requirements e CI installano torch+cu121 per supporto CUDA"
```

**Failure modes:**
- CI build fail con conflitto torch: assicurarsi che `torch --extra-index-url ...` venga PRIMA dell'install di whisper (whisper specifica torch nel suo setup).

---

### Checkpoint Fase A — milestone shippabile

A questo punto l'app:
- ✅ Rileva CUDA automaticamente
- ✅ Sposta whisper + pyannote su GPU se presente
- ✅ Mostra device nella UI
- ✅ Funziona uguale su CPU (fallback)

**È già un valore concreto per Mirko** (5–10× su T1000). Se la Fase B fosse troppo rischiosa, si potrebbe shippare ORA solo la A. Procediamo comunque con B.

---

## FASE B — Pipeline di pulizia audio

### Task B1: Installare deepfilternet e clearvoice

**Files:**
- Modify: `requirements.txt`

- [ ] **Step B1.1: Tentare l'install (può richiedere build wheel)**

Run:
```
.\.venv\Scripts\python.exe -m pip install deepfilternet clearvoice
```
Expected: installazione completata. DeepFilterNet ha un nucleo Rust, fornisce wheel precompilate per Windows; ClearerVoice è pure Python.

- [ ] **Step B1.2: Verificare import**

Run:
```
.\.venv\Scripts\python.exe -c "from df.enhance import enhance, init_df; from clearvoice import ClearVoice; print('imports OK')"
```
Expected: `imports OK`

- [ ] **Step B1.3: Aggiungere a requirements.txt**

In coda a `requirements.txt`:
```
deepfilternet
clearvoice
sounddevice
```

- [ ] **Step B1.4: Commit**

```
git add requirements.txt
git commit -m "v3: dipendenze deepfilternet + clearvoice + sounddevice"
```

**Failure modes:**
- `error: Microsoft Visual C++ 14.0 or greater is required`: serve Build Tools per VS. Più probabile: usare la wheel precompilata. Provare `pip install deepfilternet --only-binary :all:`.
- ClearerVoice scarica modelli al primo uso; va bene (li bundleremo dopo).

---

### Task B2: Audio loading utility (riusa whisper.audio)

**Files:**
- Modify: `sbobinator.py` (aggiunge `load_audio_array`)
- Create: `tests/test_audio_io.py`

- [ ] **Step B2.1: Scrivere test (deve fallire)**

`tests/test_audio_io.py`:
```python
import os
import numpy as np
import subprocess
import tempfile
import sbobinator


def _make_test_wav(path, duration_s=3, freq=440, sr=16000):
    """Genera un sine con ffmpeg per testare."""
    subprocess.run([
        os.path.join(os.path.dirname(sbobinator.__file__), "dist", "ffmpeg.exe"),
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
        # sine normalizzato deve stare in [-1, 1]
        assert -1.01 < audio.min() < -0.5
        assert 0.5 < audio.max() < 1.01
```

- [ ] **Step B2.2: Eseguire (fallisce)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_audio_io.py -v
```
Expected: FAIL `no attribute 'load_audio_array'`

- [ ] **Step B2.3: Implementare in sbobinator.py**

Aggiungere dopo `get_audio_duration`:
```python
def load_audio_array(audio_path):
    """Carica un file audio come numpy float32 mono @ 16 kHz usando ffmpeg.
    Wrapper diretto di whisper.audio.load_audio per uniformità con la pipeline."""
    import whisper.audio
    return whisper.audio.load_audio(audio_path)
```

- [ ] **Step B2.4: Eseguire (passa)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_audio_io.py -v
```
Expected: 2 passed

- [ ] **Step B2.5: Commit**

```
git add sbobinator.py tests/test_audio_io.py
git commit -m "v3: load_audio_array (wrapper di whisper.audio) + test"
```

**Failure modes:**
- ffmpeg non trovato: il test usa `dist/ffmpeg.exe`. Se mancante, copiarlo da una build precedente.

---

### Task B3: denoise_df3 function

**Files:**
- Modify: `sbobinator.py`
- Create: `tests/test_preprocess.py` (con marker slow)

- [ ] **Step B3.1: Scrivere test (slow)**

`tests/test_preprocess.py`:
```python
import os
import numpy as np
import subprocess
import tempfile
import pytest
import sbobinator


def _make_noisy_wav(path, sr=16000, duration_s=3):
    """Genera sine + rumore con ffmpeg."""
    subprocess.run([
        os.path.join(os.path.dirname(sbobinator.__file__), "dist", "ffmpeg.exe"),
        "-f", "lavfi", "-i",
        f"sine=frequency=440:duration={duration_s},anoisesrc=color=white:duration={duration_s}:amplitude=0.1",
        "-filter_complex", "[0:0][1:0]amix=inputs=2:duration=longest",
        "-ac", "1", "-ar", str(sr), "-y", path
    ], check=True, capture_output=True)


@pytest.mark.slow
def test_denoise_df3_returns_same_shape():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "noisy.wav")
        _make_noisy_wav(wav)
        audio = sbobinator.load_audio_array(wav)
        cleaned = sbobinator.denoise_df3(audio, sr=16000)
        assert isinstance(cleaned, np.ndarray)
        assert cleaned.dtype == np.float32
        assert cleaned.shape == audio.shape
        assert not np.isnan(cleaned).any()


@pytest.mark.slow
def test_denoise_df3_reduces_noise():
    """Il rumore RMS deve scendere dopo denoise (verifica che faccia qualcosa)."""
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "noisy.wav")
        _make_noisy_wav(wav)
        audio = sbobinator.load_audio_array(wav)
        cleaned = sbobinator.denoise_df3(audio, sr=16000)
        # Su un sine + rumore, il segnale pulito ha RMS più basso del rumore originale
        # nelle zone non-vocali. Test conservativo: RMS_cleaned < RMS_original * 1.1
        rms_orig = np.sqrt(np.mean(audio ** 2))
        rms_cleaned = np.sqrt(np.mean(cleaned ** 2))
        assert rms_cleaned < rms_orig * 1.5, f"RMS non ridotto: {rms_orig} -> {rms_cleaned}"
```

- [ ] **Step B3.2: Eseguire (fallisce)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py -v -m slow
```
Expected: FAIL `no attribute 'denoise_df3'`

- [ ] **Step B3.3: Implementare denoise_df3 in sbobinator.py**

Aggiungere dopo `load_audio_array`:
```python
_df_state = None  # cache del modello DF3 per evitare reload ad ogni chiamata


def denoise_df3(audio, sr=16000):
    """Riduce il rumore di fondo con DeepFilterNet 3.
    audio: numpy float32 mono @ sr.
    Ritorna numpy float32 mono @ sr (stessa shape).
    """
    global _df_state
    import torch
    from df.enhance import enhance, init_df

    if _df_state is None:
        _df_state = init_df(log_level="ERROR")
    model, df_state, _ = _df_state

    # df vuole tensor torch (1, T) su CPU; sposta su device se possibile
    device = get_device()
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    if device == "cuda":
        audio_t = audio_t.cuda()
        if hasattr(model, "to"):
            model = model.to("cuda")

    with torch.no_grad():
        cleaned = enhance(model, df_state, audio_t)
    return cleaned.squeeze(0).cpu().numpy().astype("float32")
```

- [ ] **Step B3.4: Eseguire (passa)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_denoise_df3_returns_same_shape tests/test_preprocess.py::test_denoise_df3_reduces_noise -v -m slow
```
Expected: 2 passed (può richiedere 10-30s, modello scaricato al primo run).

- [ ] **Step B3.5: Commit**

```
git add sbobinator.py tests/test_preprocess.py
git commit -m "v3: denoise_df3 con DeepFilterNet 3 + test integration"
```

**Failure modes:**
- `init_df` fallisce per modello assente: DF scarica al primo uso. Verificare connessione.
- `out of memory` su CUDA: T1000 ha 4 GB, df è piccolo (~50 MB). Se in produzione succede, audio troppo lungo: spezzare in chunk.
- API df.enhance cambiata: in caso, leggere `df.__version__` e adeguare la chiamata.

---

### Task B4: enhance_clearvoice (MossFormer2 SE)

**Files:**
- Modify: `sbobinator.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step B4.1: Aggiungere test (slow)**

In `tests/test_preprocess.py`, aggiungere:
```python
@pytest.mark.slow
def test_enhance_clearvoice_returns_same_shape():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "noisy.wav")
        _make_noisy_wav(wav)
        audio = sbobinator.load_audio_array(wav)
        cleaned = sbobinator.enhance_clearvoice(audio, sr=16000)
        assert isinstance(cleaned, np.ndarray)
        assert cleaned.dtype == np.float32
        # ClearVoice opera a 16/48 kHz internamente; verifichiamo solo che esca audio sensato
        assert len(cleaned) > 0
        assert not np.isnan(cleaned).any()
```

- [ ] **Step B4.2: Eseguire (fallisce)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_enhance_clearvoice_returns_same_shape -v -m slow
```
Expected: FAIL `no attribute 'enhance_clearvoice'`

- [ ] **Step B4.3: Implementare in sbobinator.py**

Aggiungere dopo `denoise_df3`:
```python
_cv_se = None


def enhance_clearvoice(audio, sr=16000):
    """Speech enhancement (denoising avanzato) con ClearerVoice MossFormer2 SE.
    audio: numpy float32 mono @ sr (16 kHz). Ritorna numpy float32 mono @ 16 kHz.
    """
    global _cv_se
    import numpy as np
    from clearvoice import ClearVoice

    if _cv_se is None:
        _cv_se = ClearVoice(task="speech_enhancement",
                            model_names=["MossFormer2_SE_48K"])
    # ClearVoice accetta path; salviamo temporaneo. Le librerie audio dell'app
    # ci sono già: scipy o soundfile dovrebbero essere disponibili.
    import tempfile, os
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.wav")
        sf.write(in_path, audio, sr)
        result = _cv_se(input_path=in_path, online_write=False)
        # ClearVoice ritorna dict o array a seconda della versione; gestiamo entrambi
        if isinstance(result, dict):
            cleaned = list(result.values())[0]
        else:
            cleaned = result
        if hasattr(cleaned, "numpy"):
            cleaned = cleaned.numpy()
        return np.asarray(cleaned, dtype="float32").squeeze()
```

- [ ] **Step B4.4: Eseguire (passa)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_enhance_clearvoice_returns_same_shape -v -m slow
```
Expected: 1 passed (può richiedere 30-60s, scarica modello al primo run)

- [ ] **Step B4.5: Commit**

```
git add sbobinator.py tests/test_preprocess.py
git commit -m "v3: enhance_clearvoice con MossFormer2_SE_48K + test"
```

**Failure modes:**
- `ModuleNotFoundError: soundfile`: `pip install soundfile`.
- API ClearVoice cambiata: la libreria è nuova, controlla README su GitHub di modelscope/ClearerVoice-Studio.
- Tempo lungo: il modello SE 48K è ~50 MB ma carica la prima volta. Cache HF.

---

### Task B5: separate_clearvoice (MossFormer2 SS)

**Files:**
- Modify: `sbobinator.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step B5.1: Aggiungere test**

In `tests/test_preprocess.py`:
```python
@pytest.mark.slow
def test_separate_clearvoice_returns_streams():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "noisy.wav")
        _make_noisy_wav(wav)
        audio = sbobinator.load_audio_array(wav)
        streams = sbobinator.separate_clearvoice(audio, sr=16000)
        assert isinstance(streams, list)
        assert len(streams) >= 1
        for s in streams:
            assert isinstance(s, np.ndarray)
            assert s.dtype == np.float32
            assert not np.isnan(s).any()
```

- [ ] **Step B5.2: Eseguire (fallisce)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_separate_clearvoice_returns_streams -v -m slow
```
Expected: FAIL

- [ ] **Step B5.3: Implementare**

Aggiungere dopo `enhance_clearvoice`:
```python
_cv_ss = None


def separate_clearvoice(audio, sr=16000):
    """Speech separation con ClearerVoice MossFormer2 SS.
    Ritorna lista di numpy float32 (uno per voce separata).
    """
    global _cv_ss
    import numpy as np
    from clearvoice import ClearVoice

    if _cv_ss is None:
        _cv_ss = ClearVoice(task="speech_separation",
                            model_names=["MossFormer2_SS_16K"])
    import tempfile, os
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.wav")
        sf.write(in_path, audio, sr)
        result = _cv_ss(input_path=in_path, online_write=False)
        if isinstance(result, dict):
            streams = list(result.values())
        elif isinstance(result, (list, tuple)):
            streams = list(result)
        else:
            streams = [result]
        out = []
        for s in streams:
            if hasattr(s, "numpy"):
                s = s.numpy()
            out.append(np.asarray(s, dtype="float32").squeeze())
        return out
```

- [ ] **Step B5.4: Eseguire (passa)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_separate_clearvoice_returns_streams -v -m slow
```
Expected: 1 passed

- [ ] **Step B5.5: Commit**

```
git add sbobinator.py tests/test_preprocess.py
git commit -m "v3: separate_clearvoice con MossFormer2_SS_16K + test"
```

---

### Task B6: super_resolve (ClearerVoice SR)

**Files:**
- Modify: `sbobinator.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step B6.1: Aggiungere test**

In `tests/test_preprocess.py`:
```python
@pytest.mark.slow
def test_super_resolve_upsamples():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "low.wav")
        # genera audio a 16 kHz, super-res deve portarlo a 48 kHz
        _make_noisy_wav(wav, sr=16000, duration_s=2)
        audio = sbobinator.load_audio_array(wav)
        upsampled, new_sr = sbobinator.super_resolve(audio, sr=16000)
        assert new_sr == 48000
        # ratio durata deve essere ~3× (48/16)
        ratio = len(upsampled) / len(audio)
        assert 2.8 < ratio < 3.2
```

- [ ] **Step B6.2: Eseguire (fallisce)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_super_resolve_upsamples -v -m slow
```
Expected: FAIL

- [ ] **Step B6.3: Implementare**

Aggiungere dopo `separate_clearvoice`:
```python
_cv_sr = None


def super_resolve(audio, sr=16000):
    """Super-resolution: porta l'audio a 48 kHz con ClearerVoice SR.
    Ritorna (audio_48k, 48000).
    """
    global _cv_sr
    import numpy as np
    from clearvoice import ClearVoice

    if _cv_sr is None:
        _cv_sr = ClearVoice(task="speech_super_resolution",
                            model_names=["MossFormer2_SR_48K"])
    import tempfile, os
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.wav")
        sf.write(in_path, audio, sr)
        result = _cv_sr(input_path=in_path, online_write=False)
        if isinstance(result, dict):
            up = list(result.values())[0]
        else:
            up = result
        if hasattr(up, "numpy"):
            up = up.numpy()
        return np.asarray(up, dtype="float32").squeeze(), 48000
```

- [ ] **Step B6.4: Eseguire (passa)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_super_resolve_upsamples -v -m slow
```
Expected: 1 passed

- [ ] **Step B6.5: Commit**

```
git add sbobinator.py tests/test_preprocess.py
git commit -m "v3: super_resolve con MossFormer2_SR_48K + test"
```

---

### Task B7: preprocess_audio orchestrator

**Files:**
- Modify: `sbobinator.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step B7.1: Test**

In `tests/test_preprocess.py`:
```python
@pytest.mark.slow
def test_preprocess_audio_no_flags_returns_path():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "in.wav")
        _make_noisy_wav(wav)
        out_path, sr = sbobinator.preprocess_audio(wav)
        # senza flag, restituisce path originale
        assert out_path == wav
        assert sr == 16000


@pytest.mark.slow
def test_preprocess_audio_denoise_writes_new_file():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "in.wav")
        _make_noisy_wav(wav)
        out_path, sr = sbobinator.preprocess_audio(wav, denoise=True)
        assert out_path != wav
        assert os.path.isfile(out_path)
        assert sr == 16000
        # il file pulito deve esistere ed essere caricabile
        cleaned = sbobinator.load_audio_array(out_path)
        assert not np.isnan(cleaned).any()
```

- [ ] **Step B7.2: Eseguire (fallisce)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_preprocess_audio_no_flags_returns_path tests/test_preprocess.py::test_preprocess_audio_denoise_writes_new_file -v -m slow
```
Expected: FAIL

- [ ] **Step B7.3: Implementare**

Aggiungere dopo `super_resolve`:
```python
def preprocess_audio(audio_path, denoise=False, denoise_engine="df3",
                    separate=False, superres=False, on_status=None):
    """Pipeline di pulizia audio. Restituisce (path_processato, sample_rate).
    Se nessun flag attivo, ritorna (audio_path, 16000) senza toccare il file.
    Se flag attivi, scrive WAV temporanei accanto all'audio originale con suffissi.
    """
    if not (denoise or separate or superres):
        return audio_path, 16000

    import os
    import soundfile as sf
    import numpy as np

    base, _ = os.path.splitext(audio_path)
    audio = load_audio_array(audio_path)
    sr = 16000

    if denoise:
        if on_status:
            on_status(f"Pulizia rumore ({denoise_engine})...")
        if denoise_engine == "mossformer":
            audio = enhance_clearvoice(audio, sr=sr)
        else:
            audio = denoise_df3(audio, sr=sr)

    if separate:
        if on_status:
            on_status("Separazione voci sovrapposte...")
        streams = separate_clearvoice(audio, sr=sr)
        # Per ora prendiamo lo stream più "energico" (RMS più alto) come voce principale.
        # In una versione avanzata trascriveremmo tutti gli stream separatamente.
        audio = max(streams, key=lambda s: float(np.sqrt(np.mean(s ** 2))))

    if superres:
        if on_status:
            on_status("Super-risoluzione audio...")
        audio, sr = super_resolve(audio, sr=sr)

    out_path = base + ".cleaned.wav"
    sf.write(out_path, audio, sr)
    return out_path, sr
```

- [ ] **Step B7.4: Eseguire (passa)**

Run:
```
.\.venv\Scripts\python.exe -m pytest tests/test_preprocess.py::test_preprocess_audio_no_flags_returns_path tests/test_preprocess.py::test_preprocess_audio_denoise_writes_new_file -v -m slow
```
Expected: 2 passed

- [ ] **Step B7.5: Commit**

```
git add sbobinator.py tests/test_preprocess.py
git commit -m "v3: preprocess_audio orchestrator + test"
```

---

### Task B8: Integrare preprocess in transcribe()

**Files:**
- Modify: `sbobinator.py`

- [ ] **Step B8.1: Aggiungere parametri a transcribe()**

Trovare in `sbobinator.py` la signature di `transcribe`:
```python
def transcribe(audio_path, model_name, ui_callbacks, resume_from=0.0,
               diarize_on=False, num_speakers=None):
```

Sostituire con:
```python
def transcribe(audio_path, model_name, ui_callbacks, resume_from=0.0,
               diarize_on=False, num_speakers=None,
               denoise_on=False, denoise_engine="df3",
               separate_on=False, superres_on=False):
```

- [ ] **Step B8.2: Inserire chiamata a preprocess prima del caricamento del modello whisper**

In `transcribe`, trovare la riga:
```python
    on_status("Caricamento modello in memoria...")
```

Subito PRIMA di quella riga, inserire:
```python
    # Pipeline di pulizia audio opzionale
    if denoise_on or separate_on or superres_on:
        try:
            audio_path, _sr = preprocess_audio(
                audio_path,
                denoise=denoise_on,
                denoise_engine=denoise_engine,
                separate=separate_on,
                superres=superres_on,
                on_status=on_status,
            )
        except Exception as e:
            on_status(f"Pulizia audio fallita ({e}); proseguo con l'audio originale.")
```

- [ ] **Step B8.3: Sintassi OK**

Run:
```
.\.venv\Scripts\python.exe -m py_compile sbobinator.py
```
Expected: nessun output

- [ ] **Step B8.4: Commit**

```
git add sbobinator.py
git commit -m "v3: integra preprocess_audio nel flusso transcribe"
```

---

### Task B9: UI — profili qualità + spunte avanzate

**Files:**
- Modify: `sbobinator.py` (App.__init__, start_transcription, thread args)

- [ ] **Step B9.1: Aggiungere controlli UI (radio profilo + advanced)**

Trovare in `sbobinator.py`, dentro `App.__init__`, la riga del `diar_row` (riga della casella "Riconosci chi parla"). Subito DOPO `diar_row.pack(fill="x", pady=(0, 10))`, aggiungere:

```python
        profile_row = tk.Frame(frame, bg=CARA_BLU)
        profile_row.pack(fill="x", pady=(0, 8))
        tk.Label(profile_row, text="Profilo:", font=("Arial", 10),
                 fg=CARA_TXT, bg=CARA_BLU).pack(side="left")
        self.profile_var = tk.StringVar(value="standard")
        for value, label in [("standard", "Standard"),
                             ("intercettazione", "🚓 Intercettazione"),
                             ("max", "🎯 Max qualità"),
                             ("advanced", "⚙️ Avanzato")]:
            tk.Radiobutton(profile_row, text=label, variable=self.profile_var,
                           value=value, font=("Arial", 9), bg=CARA_BLU, fg=CARA_TXT,
                           selectcolor=CARA_BLU_SCURO, activebackground=CARA_BLU,
                           activeforeground=CARA_ORO, highlightthickness=0,
                           command=self._on_profile_change).pack(side="left", padx=(8, 0))

        adv_row = tk.Frame(frame, bg=CARA_BLU)
        adv_row.pack(fill="x", pady=(0, 8))
        self.denoise_var = tk.BooleanVar(value=False)
        self.separate_var = tk.BooleanVar(value=False)
        self.superres_var = tk.BooleanVar(value=False)
        for var, text in [(self.denoise_var, "Denoise"),
                          (self.separate_var, "Separa voci"),
                          (self.superres_var, "Super-res")]:
            tk.Checkbutton(adv_row, text=text, variable=var, font=("Arial", 9),
                           bg=CARA_BLU, fg=CARA_TXT, selectcolor=CARA_BLU_SCURO,
                           activebackground=CARA_BLU, activeforeground=CARA_ORO,
                           highlightthickness=0).pack(side="left", padx=(10, 0))
        self.adv_row = adv_row
```

- [ ] **Step B9.2: Aggiungere metodo _on_profile_change**

Dentro la classe App, dopo `rename_speakers`, aggiungere:
```python
    def _on_profile_change(self):
        p = self.profile_var.get()
        if p == "standard":
            self.denoise_var.set(False)
            self.separate_var.set(False)
            self.superres_var.set(False)
        elif p == "intercettazione":
            self.denoise_var.set(True)
            self.separate_var.set(True)
            self.superres_var.set(False)
        elif p == "max":
            self.denoise_var.set(True)
            self.separate_var.set(True)
            self.superres_var.set(True)
        # advanced: lascia le spunte come sono (l'utente le tocca a mano)
```

- [ ] **Step B9.3: Passare i parametri al thread transcribe**

Trovare in `sbobinator.py`, dentro `start_transcription`, la riga:
```python
        t = threading.Thread(target=transcribe,
                             args=(path, model_name, callbacks, resume_from, diarize_on, num_speakers),
                             daemon=True)
```

Sostituire con:
```python
        denoise_on = self.denoise_var.get()
        separate_on = self.separate_var.get()
        superres_on = self.superres_var.get()
        engine = "mossformer" if self.profile_var.get() == "max" else "df3"
        t = threading.Thread(
            target=transcribe,
            args=(path, model_name, callbacks, resume_from,
                  diarize_on, num_speakers,
                  denoise_on, engine, separate_on, superres_on),
            daemon=True,
        )
```

- [ ] **Step B9.4: Test GUI headless**

Run:
```
.\.venv\Scripts\python.exe -c "import tkinter,sbobinator; r=tkinter.Tk(); r.withdraw(); a=sbobinator.App(r); a.profile_var.set('intercettazione'); a._on_profile_change(); assert a.denoise_var.get() and a.separate_var.get() and not a.superres_var.get(); print('GUI OK'); r.destroy()"
```
Expected: `GUI OK`

- [ ] **Step B9.5: Commit**

```
git add sbobinator.py
git commit -m "v3: UI con profili qualità (Standard/Intercettazione/Max/Avanzato)"
```

---

### Task B10: CLI --selftest per validazione exe frozen

**Files:**
- Modify: `sbobinator.py` (main)

- [ ] **Step B10.1: Aggiungere branch CLI in main()**

Trovare in `sbobinator.py` la funzione `main()`. All'inizio (subito dopo `def main():`), aggiungere:
```python
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        # Diagnostica per exe frozen: prova denoise + diarize su un audio
        # e scrive log accanto al file. Exit 0 se tutto ok.
        if len(sys.argv) < 3:
            sys.stderr = sys.__stderr__ or sys.stderr
            print("uso: Sbobinator.exe --selftest <audio>", file=sys.stderr)
            sys.exit(2)
        audio = sys.argv[2]
        log_path = os.path.splitext(audio)[0] + ".selftest.log"
        with open(log_path, "w", encoding="utf-8") as log:
            def w(msg):
                log.write(msg + "\n"); log.flush()
            try:
                w(f"device: {get_device_info()}")
                w("loading audio...")
                a = load_audio_array(audio)
                w(f"audio len: {len(a)} samples")
                w("denoise (df3)...")
                cleaned = denoise_df3(a)
                w(f"cleaned len: {len(cleaned)}")
                w("diarize...")
                turns = diarize(audio, num_speakers=None)
                w(f"turns: {len(turns)}")
                w("SELFTEST OK")
                sys.exit(0)
            except Exception as e:
                import traceback
                w(f"SELFTEST FAIL: {e}")
                w(traceback.format_exc())
                sys.exit(1)
```

- [ ] **Step B10.2: Test rapido (con clip dal monologo)**

Run (su clip già disponibile o quella nuova quando arriva):
```
.\.venv\Scripts\python.exe -c "import sys; sys.argv = ['Sbobinator', '--selftest', 'C:/Users/Omacatl/Desktop/risposta mirko.ogg']; import sbobinator; sbobinator.main()"
```
Expected: exit 0, file `risposta mirko.selftest.log` accanto con righe "SELFTEST OK".

- [ ] **Step B10.3: Commit**

```
git add sbobinator.py
git commit -m "v3: CLI --selftest per validare exe frozen su PC target"
```

---

### Task B11: Update .spec per bundle DF3 + ClearVoice

**Files:**
- Modify: `sbobinator.spec`

- [ ] **Step B11.1: Aggiungere collect_all per i nuovi pacchetti**

In `sbobinator.spec`, trovare la lista dei pacchetti dentro il `for pkg in [...]:`. Aggiungere alla lista:
```python
'deepfilternet', 'df', 'clearvoice', 'modelscope',
'soundfile', 'sounddevice',
```

(quindi la lista diventa l'unione di quelli esistenti più questi).

- [ ] **Step B11.2: Sintassi check del .spec (parse YAML/python)**

Run:
```
.\.venv\Scripts\python.exe -c "exec(open('sbobinator.spec').read())"
```
Expected: nessun errore (script non genera output, ma non deve crashare).

Nota: lo `.spec` ha riferimenti a oggetti PyInstaller (Analysis, EXE) che non esistono nel namespace Python normale; l'`exec` può fallire con `NameError: Analysis`. Va bene: stiamo solo verificando che il file parsi.

- [ ] **Step B11.3: Commit**

```
git add sbobinator.spec
git commit -m "v3: .spec raccoglie deepfilternet, clearvoice, soundfile, sounddevice"
```

---

### Task B12: Build locale + smoke test exe

**Files:**
- (Nessun file — operazione di build)

- [ ] **Step B12.1: Pulire eventuali processi Sbobinator running (lock dell'exe)**

Run:
```
Get-Process -Name Sbobinator -ErrorAction SilentlyContinue | Stop-Process -Force
```

- [ ] **Step B12.2: Build con --clean su dist_v3**

Run:
```
.\.venv\Scripts\pyinstaller.exe sbobinator.spec --noconfirm --clean --distpath dist_v3 --workpath build_v3
```
Expected: build completa, exit 0. L'exe pesa ~600-800 MB con CUDA.

- [ ] **Step B12.3: Verifica bundling dei modelli DF3 e ClearVoice**

Run:
```
.\.venv\Scripts\python.exe -m PyInstaller.utils.cliutils.archive_viewer -l dist_v3\Sbobinator.exe > archive_v3.txt
```
Poi:
```
.\.venv\Scripts\python.exe -c "import re; t=open('archive_v3.txt').read(); print('df model:', bool(re.search(r'DeepFilterNet', t))); print('clearvoice:', bool(re.search(r'clearvoice', t, re.I))); print('cuda:', bool(re.search(r'cudart|cublas', t, re.I)))"
```
Expected: i tre check `True`.

- [ ] **Step B12.4: Setup dist_v3 per test (ffmpeg + modelli whisper)**

Run:
```
Copy-Item dist\ffmpeg.exe dist_v3\ffmpeg.exe -Force
if (-not (Test-Path dist_v3\models)) { New-Item -ItemType Junction -Path dist_v3\models -Target $PSScriptRoot\dist\models | Out-Null }
```

- [ ] **Step B12.5: Smoke test exe (`--selftest`)**

Run:
```
.\dist_v3\Sbobinator.exe --selftest "C:\Users\Omacatl\Desktop\risposta mirko.ogg"
$exit = $LASTEXITCODE
"selftest exit: $exit"
Get-Content "C:\Users\Omacatl\Desktop\risposta mirko.selftest.log" -Tail 5
```
Expected: exit 0, log con "SELFTEST OK".

- [ ] **Step B12.6: (Solo se exit != 0) diagnosi**

Se il selftest fallisce, leggere il log completo:
```
Get-Content "C:\Users\Omacatl\Desktop\risposta mirko.selftest.log"
```

Failure modes comuni:
- `ModuleNotFoundError: df`: il modulo `df` di DeepFilterNet non è stato bundlato. Aggiungere `'df'` agli hiddenimports espliciti nella spec.
- `FileNotFoundError: cudart64_*.dll`: CUDA runtime non bundlato. Aggiungere `--collect-binaries torch` o spostare DLL manualmente. (In dev senza CUDA reale non si vede.)
- `clearvoice not found`: i modelli ClearVoice si scaricano al primo uso. Su PC offline serve pre-popolare la cache (Task B13).

- [ ] **Step B12.7: Commit (solo se passa)**

Niente da committare (build), ma documentare lo stato:
```
git status
```

---

### Task B13: (Condizionale) Pre-popolare cache ClearVoice nel bundle

**Files:**
- Create: `cv_models/` (directory locale con modelli pre-scaricati)
- Modify: `sbobinator.spec`

Da fare SOLO se Task B12.5 fallisce con errori di download al runtime.

- [ ] **Step B13.1: Identificare i modelli scaricati da ClearVoice**

Run:
```
.\.venv\Scripts\python.exe -c "from clearvoice import ClearVoice; cv = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K']); import os; print('cache:', os.path.expanduser('~/.cache/clearvoice'))"
```
Expected: stampa il path della cache. Verificare manualmente che contenga i file.

- [ ] **Step B13.2: Copiare la cache nel repo come cv_models/**

Run:
```
$src = "$env:USERPROFILE\.cache\clearvoice"
if (Test-Path $src) { Copy-Item $src cv_models -Recurse -Force; "cv_models popolato" } else { "cache vuota, controlla manualmente" }
```

- [ ] **Step B13.3: Aggiungere cv_models al bundle nello .spec**

In `sbobinator.spec`, dentro la lista `datas`, aggiungere:
```python
datas += [('cv_models', 'cv_models')]
```

- [ ] **Step B13.4: Far puntare ClearVoice a cv_models in frozen**

In `sbobinator.py`, nel guard iniziale (vicino a HF_HOME), aggiungere dopo il blocco HF:
```python
if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", get_base_path())
    _cv = os.path.join(_base, "cv_models")
    if os.path.isdir(_cv):
        os.environ.setdefault("CLEARVOICE_CACHE_DIR", _cv)
```

- [ ] **Step B13.5: Rebuild + selftest**

Ripetere Task B12.

- [ ] **Step B13.6: Commit**

```
git add cv_models sbobinator.spec sbobinator.py
git commit -m "v3: pre-popola cache ClearVoice nel bundle (offline-ready)"
```

---

## FASE C — End-to-end test su intercettazione reale

### Task C1: Script di validazione manuale

**Files:**
- Create: `tests/manual/run_pipeline.py`

- [ ] **Step C1.1: Scrivere lo script**

`tests/manual/run_pipeline.py`:
```python
"""End-to-end manuale: dato un audio, esegue trascrizione con varie configurazioni
e produce un report di confronto. NON è un pytest — si lancia a mano.

Uso: python tests/manual/run_pipeline.py <audio_path>
"""
import sys
import os
import time

sys.stdout = None
sys.stderr = None
os.environ["PATH"] = os.path.join(os.path.dirname(__file__), "..", "..", "dist") + os.pathsep + os.environ.get("PATH", "")

import sbobinator
sys.stderr = sys.__stderr__

audio = sys.argv[1]
out_real = sys.__stdout__


def run(label, **kwargs):
    txt_path = os.path.splitext(audio)[0] + f".{label}.txt"
    if os.path.exists(txt_path):
        os.remove(txt_path)
    # forziamo l'app a scrivere lì
    # Cambiamo audio_path per ogni run? In realtà l'app scrive accanto al sorgente:
    # facciamo una copia con nome diverso per separare gli output
    src_dir = os.path.dirname(audio)
    src_base = os.path.basename(audio)
    name, ext = os.path.splitext(src_base)
    copy_path = os.path.join(src_dir, f"{name}_{label}{ext}")
    if not os.path.exists(copy_path):
        import shutil
        shutil.copy(audio, copy_path)
    out_txt = os.path.splitext(copy_path)[0] + ".txt"
    if os.path.exists(out_txt):
        os.remove(out_txt)

    done = {}
    segs = []

    def on_seg(line):
        if line != "__RELOAD__":
            segs.append(line)

    t0 = time.time()
    sbobinator.transcribe(
        copy_path, "medium",
        (lambda m: None, lambda m, p: None, on_seg,
         lambda ok, r, i: done.update(ok=ok, r=r, i=i)),
        **kwargs,
    )
    elapsed = time.time() - t0
    print(f"\n=== {label} ({elapsed:.1f}s) ===", file=out_real)
    print(f"success: {done.get('ok')} | segments: {len(segs)}", file=out_real)
    if os.path.exists(out_txt):
        with open(out_txt, encoding="utf-8") as f:
            for line in f.read().splitlines()[:5]:
                print(" ", line, file=out_real)


# Confronto: standard / con denoise / con denoise+separa / max
print(f"AUDIO: {audio}", file=out_real)
print(f"device: {sbobinator.get_device_info()}", file=out_real)
run("baseline")
run("denoise", denoise_on=True)
run("denoise_separate", denoise_on=True, separate_on=True)
run("max", denoise_on=True, denoise_engine="mossformer", separate_on=True, superres_on=True)
```

- [ ] **Step C1.2: Quando arriva l'intercettazione, lanciarla**

Run (sostituire il path):
```
New-Item -ItemType Directory -Force -Path tests\manual | Out-Null
.\.venv\Scripts\python.exe tests\manual\run_pipeline.py "C:\Users\Omacatl\Desktop\<intercettazione>.<ext>"
```
Expected: stampa 4 sezioni (baseline, denoise, denoise_separate, max) con il tempo di elaborazione e le prime 5 righe della trascrizione. Confrontare manualmente.

- [ ] **Step C1.3: Commit dello script (NON degli output)**

```
git add tests/manual/run_pipeline.py
git commit -m "v3: script end-to-end di confronto qualità su audio reale"
```

- [ ] **Step C1.4: Scrivere un breve report di analisi**

In `docs/superpowers/specs/2026-05-30-v3-validation-report.md`, scrivere a mano (post-run) un confronto onesto delle quattro configurazioni: quanto miglior­a la trascrizione? Quanto cresce il tempo? Quale profilo è il migliore per quel tipo di intercettazione?

```
git add docs/superpowers/specs/2026-05-30-v3-validation-report.md
git commit -m "v3: report di validazione qualità su audio reale"
```

**Failure modes:**
- Tempo di elaborazione esplosivo: senza GPU (dev PC), la pipeline è SLOW. Aspettare ~5-10 min per file da 60s. Su T1000 sarà 5-10× più rapida.
- Out of memory in dev (CPU): se l'audio è lungo (> 2 min), può saturare RAM. Pre-tagliare a 60s con ffmpeg.

---

## FASE D — Deploy

### Task D1: Smoke test finale exe

- [ ] **Step D1.1: Re-run selftest sul dist_v3**

Run:
```
.\dist_v3\Sbobinator.exe --selftest "C:\Users\Omacatl\Desktop\risposta mirko.ogg"
Get-Content "C:\Users\Omacatl\Desktop\risposta mirko.selftest.log" -Tail 8
```
Expected: SELFTEST OK.

---

### Task D2: Merge su main + push (CI rebuild + release)

- [ ] **Step D2.1: Verifica stato pulito sul branch**

Run:
```
git status --short
```
Expected: tutto committato. Untracked solo dist_v3, build_v3, archive_v3.txt (non da committare).

- [ ] **Step D2.2: Aggiungere dist_v3, build_v3 a .gitignore**

In `.gitignore`, aggiungere:
```
dist_v3/
build_v3/
archive_v3.txt
selftest.log
*.selftest.log
*.cleaned.wav
```

Run:
```
git add .gitignore
git commit -m "v3: gitignore artefatti build/test"
```

- [ ] **Step D2.3: Merge su main**

Run:
```
git checkout main
git pull --ff-only
git merge v3-quality-pipeline --no-edit
```
Expected: fast-forward merge.

- [ ] **Step D2.4: Push (avvia CI)**

Run:
```
git push origin main
```
Expected: push success. La CI parte, build con CUDA + pyannote + ClearVoice + DF3 → ~25-40 min stimati.

- [ ] **Step D2.5: Monitor CI**

Run:
```
$run_id = (Invoke-RestMethod "https://api.github.com/repos/Omacatl94/whisper-sbobinator/actions/runs?per_page=1" -Headers @{'User-Agent'='claude'}).workflow_runs[0].id
"run id: $run_id"
```

Poi attendere o monitorare via web. Quando completata, verificare:
```
$rel = Invoke-RestMethod "https://api.github.com/repos/Omacatl94/whisper-sbobinator/releases/tags/v1.0" -Headers @{'User-Agent'='claude'}
$asset = $rel.assets | Where-Object { $_.name -eq 'Sbobinator.zip' }
"size MB: $([math]::Round($asset.size / 1MB, 0))"
"updated: $($asset.updated_at)"
```
Expected: zip aggiornato, size ~ 700-1000 MB (con CUDA è il salto principale).

---

## Self-review del piano

**Spec coverage:**
- ✅ Fase A: get_device, integrazione whisper, integrazione pyannote, UI indicator, install CUDA, requirements + CI
- ✅ Fase B: deps install, audio loader, df3 denoise, clearvoice enhance/separate/super-res, orchestrator preprocess, integrazione transcribe, UI profili, --selftest, .spec, build, condizionale cache ClearVoice
- ✅ Fase C: script end-to-end e report
- ✅ Fase D: deploy
- ✅ Strategia test/debug: documentata up-front, riferimenti in ogni Task

**Gap dichiarati e fuori scope:**
- Click-to-replay + correzione (Fase D originale): rinviata a v3.1, plan separato.
- Batch processing: rinviata a v3.1, plan separato.
- TSE (Target Speaker Extraction): rinviata a v4.
- Italian fine-tune Whisper Large V3 It Cv16: valutazione manuale dopo i risultati Fase C; può diventare un'opzione "modello = italiano" nell'UI senza grosse modifiche.

**Type consistency:**
- `get_device()` -> str ("cuda" | "cpu") usato ovunque.
- `get_device_info()` -> dict con keys "device", "name", "vram_gb".
- `load_audio_array(path)` -> np.ndarray float32 mono @ 16 kHz.
- `denoise_df3(audio, sr=16000)` -> np.ndarray float32 mono.
- `enhance_clearvoice(audio, sr=16000)` -> np.ndarray float32 mono @ 16 kHz.
- `separate_clearvoice(audio, sr=16000)` -> List[np.ndarray] float32.
- `super_resolve(audio, sr=16000)` -> Tuple[np.ndarray, int] (audio, 48000).
- `preprocess_audio(path, ...)` -> Tuple[str, int] (path_processato, sample_rate).
- `transcribe(...)` -> signature estesa con kwargs di preprocessing.

Tutto consistente.

**Placeholder scan:** Nessun "TBD"/"TODO"/"fill in" trovato. Tutti i blocchi di codice sono completi.
