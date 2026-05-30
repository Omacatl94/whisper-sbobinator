import sys
import os
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import warnings

warnings.filterwarnings("ignore")

# Massimizza l'uso della CPU: usa tutti i core disponibili per torch e BLAS.
# Vale per i pezzi della pipeline che girano su CPU anche quando c'e' la GPU
# (es. clustering di pyannote, alcuni step di ClearerVoice).
import os as _os
_n = _os.cpu_count() or 4
_os.environ.setdefault("OMP_NUM_THREADS", str(_n))
_os.environ.setdefault("MKL_NUM_THREADS", str(_n))
_os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n))
_os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_n))
try:
    import torch as _torch
    _torch.set_num_threads(_n)
    _torch.set_num_interop_threads(max(2, _n // 4))
except Exception:
    pass

# Workaround: clearvoice tira dentro speechbrain, che ha un LazyModule per
# k2_fsa. inspect.stack() (chiamato da pyannote/lightning) ne triggera l'import
# che fallisce perché k2 non è installato. Rendiamo il fallimento "silente"
# restituendo un modulo stub.
try:
    from speechbrain.utils import importutils as _sb_iu
    _orig_ensure = _sb_iu.LazyModule.ensure_module

    def _safe_ensure(self, stacklevel=1):
        try:
            return _orig_ensure(self, stacklevel)
        except ImportError:
            import types as _t
            stub = _t.ModuleType(self.target)
            stub.__file__ = None
            self.lazy_module = stub
            return stub

    _sb_iu.LazyModule.ensure_module = _safe_ensure
except Exception:
    pass

# In una build PyInstaller "windowed" (console=False) sys.stdout e sys.stderr
# valgono None. whisper/tqdm ci scrivono sopra durante la trascrizione e l'app
# crasha con "NoneType object has no attribute 'write'". Reindirizziamo i
# flussi mancanti su devnull così ogni scrittura va a vuoto senza errori.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# Frozen: niente più chdir/TORCH_HOME (era per ClearVoice/DNS64, droppati)

MODEL_INFO = {
    "medium": {
        "engine": "whisper",
        "label": "Whisper Medium (~1.5GB, veloce)",
        "file": "medium.pt",
        "size_gb": 1.5,
    },
    "large": {
        "engine": "whisper",
        "label": "Whisper Large-v3 (~3GB, dialetti)",
        "file": "large-v3.pt",
        "size_gb": 2.9,
    },
    "voxtral-mini": {
        "engine": "voxtral",
        "label": "Voxtral Mini 3B (~6GB, top forense)",
        "hf_id": "mistralai/Voxtral-Mini-3B-2507",
        "folder": "voxtral-mini-3b",
        "size_gb": 6.0,
    },
}

# --- Tema "Carabinieri" ---
CARA_BLU = "#0a1f44"        # blu uniforme
CARA_BLU_SCURO = "#06152e"  # blu piu' scuro (sfondi incassati)
CARA_ROSSO = "#c8102e"      # rosso banda
CARA_ORO = "#caa84a"        # oro fiamma
CARA_TXT = "#eef2f8"        # testo chiaro
COL_OK = "#36d399"
COL_WARN = "#fbbd23"
COL_ERR = "#ff6b6b"
COL_INFO = "#5fa8ff"
COL_MUTED = "#9fb0c8"


def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_models_dir():
    d = os.path.join(get_base_path(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_exists(model_name):
    info = MODEL_INFO[model_name]
    if info["engine"] == "whisper":
        return os.path.isfile(os.path.join(get_models_dir(), info["file"]))
    elif info["engine"] == "voxtral":
        # Per Voxtral controlliamo se la cartella HF cache contiene almeno il
        # config.json — segno che il modello è stato scaricato.
        folder = os.path.join(get_models_dir(), info["folder"])
        return os.path.isfile(os.path.join(folder, "config.json"))
    return False


def check_ffmpeg():
    # Cerca ffmpeg accanto all'exe (build congelata) o in dist/ (dev)
    candidates = [get_base_path(), os.path.join(get_base_path(), "dist")]
    for c in candidates:
        p = os.path.join(c, "ffmpeg.exe")
        if os.path.isfile(p):
            os.environ["PATH"] = c + os.pathsep + os.environ.get("PATH", "")
            return True
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_audio_duration(audio_path):
    try:
        import whisper.audio
        audio = whisper.audio.load_audio(audio_path)
        return len(audio) / 16000.0
    except Exception:
        return None


def load_audio_array(audio_path):
    """Carica un file audio come numpy float32 mono @ 16 kHz usando ffmpeg.
    Wrapper di whisper.audio.load_audio per uniformità nella pipeline."""
    import whisper.audio
    return whisper.audio.load_audio(audio_path)


def normalize_volume(audio_path, on_status=None):
    """Normalizzazione dinamica del volume con ffmpeg dynaudnorm.
    Alza zone con livello basso senza saturare le alte. NON modifica lo
    spettro, solo l'ampiezza nel tempo → SAFE per Whisper/ASR moderni.
    Da usare solo se l'audio originale ha grandi variazioni di livello
    (es. sussurri + parlato normale). Sull'audio già bilanciato non serve
    e può essere disattivata.
    """
    import subprocess
    import os
    base, _ = os.path.splitext(audio_path)
    out_path = base + ".normalized.wav"
    if on_status:
        on_status("Normalizzazione volume...")
    # dynaudnorm parametri standard per voce (frame 150ms, gauss 15)
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", "dynaudnorm=f=150:g=15",
        "-ar", "16000", "-ac", "1",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path, 16000


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
    """Info per UI: device + nome + VRAM disponibile (GB)."""
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


def download_model(model_name, progress_callback):
    info = MODEL_INFO[model_name]
    if info["engine"] == "voxtral":
        return _download_voxtral(model_name, progress_callback)
    return _download_whisper(model_name, progress_callback)


def _download_voxtral(model_name, progress_callback):
    """Scarica un modello Voxtral via huggingface_hub.snapshot_download."""
    info = MODEL_INFO[model_name]
    folder = os.path.join(get_models_dir(), info["folder"])
    try:
        progress_callback(f"Download Voxtral ({info['size_gb']:.1f} GB)...", -1)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=info["hf_id"],
            local_dir=folder,
            local_dir_use_symlinks=False,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def _download_whisper(model_name, progress_callback):
    import urllib.request

    urls = {
        "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    }

    info = MODEL_INFO[model_name]
    url = urls[model_name]
    dest = os.path.join(get_models_dir(), info["file"])
    dest_tmp = dest + ".downloading"

    try:
        req = urllib.request.urlopen(url)
        total = int(req.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024

        with open(dest_tmp, "wb") as f:
            while True:
                chunk = req.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    progress_callback(f"Download modello: {mb_done:.0f}/{mb_total:.0f} MB ({pct:.0f}%)", pct)
                else:
                    mb_done = downloaded / (1024 * 1024)
                    progress_callback(f"Download modello: {mb_done:.0f} MB...", -1)

        os.rename(dest_tmp, dest)
        return True, None
    except Exception as e:
        if os.path.exists(dest_tmp):
            os.remove(dest_tmp)
        return False, str(e)


def last_transcribed_time(txt_path):
    """Secondi dell'ultimo segmento salvato in un .txt (per riprendere), o 0."""
    last = 0.0
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("[") and "]" in line and " - " in line:
                    try:
                        inside = line[1:line.index("]")]
                        _, end = inside.split(" - ")
                        mm, ss = end.split(":")
                        last = int(mm) * 60 + int(ss)
                    except Exception:
                        pass
    except Exception:
        return 0.0
    return float(last)


def assign_speakers(segments, turns):
    """Assegna a ogni segmento la voce con massima sovrapposizione temporale.
    turns: lista di (start, end, label). Ritorna i segmenti con chiave 'speaker'."""
    out = []
    for seg in segments:
        best_label, best_overlap = None, 0.0
        for t_start, t_end, label in turns:
            overlap = min(seg["end"], t_end) - max(seg["start"], t_start)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        out.append({**seg, "speaker": best_label})
    return out


def speaker_label_map(segments):
    """SPEAKER_xx -> 'Interlocutore N' in ordine di prima comparsa."""
    mapping = {}
    for seg in segments:
        spk = seg.get("speaker")
        if spk and spk not in mapping:
            mapping[spk] = f"Interlocutore {len(mapping) + 1}"
    return mapping


def format_line(seg, label_map):
    """Formatta una riga: [mm:ss - mm:ss] [Etichetta: ]testo."""
    s, e = seg["start"], seg["end"]
    ts = f"[{int(s // 60):02d}:{int(s % 60):02d} - {int(e // 60):02d}:{int(e % 60):02d}]"
    text = seg["text"].strip()
    spk = seg.get("speaker")
    if spk and spk in label_map:
        return f"{ts} {label_map[spk]}: {text}"
    return f"{ts} {text}"


def diarize(audio_path, num_speakers=None, on_status=None):
    """Diarization offline su CPU. Ritorna [(start, end, label)].

    Usa pyannote 4.x. L'audio viene caricato in memoria col ffmpeg di whisper
    (evita torchcodec). I modelli sono caricati dalla cache locale (offline);
    nella build congelata stanno in 'hf_models/' dentro il bundle.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", get_base_path())
        bundled = os.path.join(base, "hf_models")
        if os.path.isdir(bundled):
            os.environ["HF_HOME"] = bundled
    if on_status:
        on_status("Riconoscimento voci...")

    import torch
    import whisper.audio
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipe.to(torch.device(get_device()))

    audio = whisper.audio.load_audio(audio_path)  # float32 mono @ 16 kHz
    waveform = torch.from_numpy(audio).unsqueeze(0)
    call_kwargs = {}
    if num_speakers:
        call_kwargs["num_speakers"] = int(num_speakers)
    dia = pipe({"waveform": waveform, "sample_rate": 16000}, **call_kwargs)
    ann = dia.speaker_diarization
    return [(t.start, t.end, label) for t, _, label in ann.itertracks(yield_label=True)]


def _transcribe_voxtral(audio_path, model_name, ui_callbacks):
    """Trascrizione con Voxtral Mini 3B (mistralai/Voxtral-Mini-3B-2507).
    Output: testo a flusso (no segmenti con timestamp puntuali).
    Per ora niente diarization (Voxtral Mini open non la include — pyannote
    da chiamare in step separato se serve)."""
    on_status, on_progress, on_segment, on_done = ui_callbacks
    import time
    info = MODEL_INFO[model_name]
    folder = os.path.join(get_models_dir(), info["folder"])

    on_status("Caricamento Voxtral in memoria...")
    on_progress("Caricamento modello...", -1)
    try:
        import torch
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        device = get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(folder)
        model = VoxtralForConditionalGeneration.from_pretrained(
            folder, torch_dtype=dtype, device_map=device,
        )
    except Exception as e:
        on_done(False, f"Errore caricamento Voxtral:\n{e}\n\nIl modello potrebbe non essere stato scaricato — usa 'Scarica modello'.", None)
        return

    on_status("Sbobinatura con Voxtral...")
    on_progress("Trascrizione...", 0)
    t0 = time.time()
    try:
        # Carichiamo audio via whisper/ffmpeg (gestisce mp3/m4a/ogg/wav)
        # e lo passiamo a Voxtral come numpy (soundfile non gestisce m4a).
        audio_np = load_audio_array(audio_path)  # float32 mono @ 16 kHz
        # Il processor vuole audio come lista + format come lista (anche per
        # un solo file). Passare numpy singolo causa "len(None)" bug.
        inputs = processor.apply_transcription_request(
            audio=[audio_np],
            model_id=info["hf_id"],
            language=["it"],
            sampling_rate=16000,
            format=["wav"],
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(device, dtype=dtype if dtype != torch.float32 else None)
        else:
            # dict di tensor
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=8000, do_sample=False)
        # Decode tutto, poi prendiamo il testo
        input_len = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs.input_ids.shape[1]
        new_tokens = outputs[:, input_len:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    except Exception as e:
        import traceback
        on_done(False, f"Errore Voxtral:\n{e}\n\n{traceback.format_exc()[:500]}", None)
        return

    elapsed = time.time() - t0
    on_status("Salvataggio trascrizione...")
    out_path = os.path.splitext(audio_path)[0] + ".txt"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        # mostriamo il testo nell'UI come una grande riga
        for line in text.split("\n"):
            if line.strip():
                on_segment(line)
    except Exception as e:
        on_done(False, f"Errore salvataggio:\n{e}", None)
        return

    mins = int(elapsed // 60); secs = int(elapsed % 60)
    on_progress("Completato!", 100)
    on_done(True, out_path, f"Voxtral — tempo: {mins}m {secs}s")


def transcribe(audio_path, model_name, ui_callbacks, resume_from=0.0,
               diarize_on=False, num_speakers=None,
               normalize_on=False):
    on_status, on_progress, on_segment, on_done = ui_callbacks

    # Normalizzazione volume opzionale (NON denoise — solo livello).
    # I modelli ASR moderni (Whisper, Voxtral) sono stati addestrati su audio
    # raw rumoroso: pre-processarli col denoise di solito PEGGIORA i risultati.
    # La normalizzazione del volume invece è "safe" perché non altera lo spettro.
    if normalize_on:
        try:
            audio_path, _ = normalize_volume(audio_path, on_status=on_status)
        except Exception as e:
            on_status(f"Normalizzazione fallita ({e}); proseguo con l'originale.")

    # Dispatch al backend appropriato in base al modello scelto
    info = MODEL_INFO.get(model_name, {})
    if info.get("engine") == "voxtral":
        return _transcribe_voxtral(audio_path, model_name, ui_callbacks)

    if not check_ffmpeg():
        on_done(False, "ffmpeg non trovato!\n\nMetti ffmpeg.exe nella stessa cartella di Sbobinator.", None)
        return

    if not model_exists(model_name):
        on_status(f"Modello {model_name} non trovato. Download in corso...")
        ok, err = download_model(model_name, lambda msg, pct: on_progress(msg, pct))
        if not ok:
            on_done(False, f"Errore download modello:\n{err}\n\nSe il PC non ha internet, scarica il modello su un altro PC e copialo nella cartella 'models/'.", None)
            return

    on_status("Caricamento modello in memoria...")
    on_progress("Caricamento modello...", -1)

    try:
        import whisper
        model = whisper.load_model(model_name, download_root=get_models_dir(), device=get_device())
    except Exception as e:
        on_done(False, f"Errore caricamento modello:\n{e}\n\nIl file potrebbe essere corrotto. Cancella la cartella 'models/' e riscarica.", None)
        return

    on_status("Analisi audio...")
    duration = get_audio_duration(audio_path)
    if duration is None:
        on_done(False, "Impossibile leggere il file audio.\n\nFormato non supportato o file corrotto.", None)
        return

    out_path = os.path.splitext(audio_path)[0] + ".txt"

    if resume_from > 0:
        on_status(f"Riprendo da {int(resume_from // 60):02d}:{int(resume_from % 60):02d}...")
    else:
        on_status("Sbobinatura in corso...")
    on_progress("Sbobinatura...", (min(99.0, resume_from / duration * 100) if duration else 0))

    start_time = time.time()

    # --- Live vera + salvataggio progressivo ---
    # Whisper non espone callback: agganciamo la sua barra interna (tqdm) per la
    # percentuale reale e catturiamo l'output di verbose=True per i segmenti, che
    # vengono mostrati E scritti su file man mano (un crash non perde il lavoro
    # fatto). Con resume_from si riparte da un punto e si accoda al file.
    # Nota: whisper.transcribe (attributo) è la FUNZIONE, non il modulo (per via
    # di "from .transcribe import transcribe" in __init__). Prendiamo il modulo
    # vero per poterne sostituire il tqdm interno.
    import importlib
    _wt = importlib.import_module("whisper.transcribe")

    frames_per_second = 100  # whisper: SAMPLE_RATE / HOP_LENGTH
    resume_frames = int(resume_from * frames_per_second)

    def _ts_to_sec(ts):
        sec = 0.0
        for part in ts.split(":"):
            sec = sec * 60 + float(part)
        return sec

    class _ProgressTqdm:
        def __init__(self, *args, total=None, **kwargs):
            self.total = total or 0
            self.n = resume_frames

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n=1):
            self.n += n
            if self.total:
                on_progress("Sbobinatura...", min(99.0, self.n / self.total * 100))

        def close(self):
            pass

    class _FakeTqdmModule:
        tqdm = _ProgressTqdm

    try:
        out_file = open(out_path, "a" if resume_from > 0 else "w", encoding="utf-8")
    except Exception as e:
        on_done(False, f"Non riesco a scrivere il file:\n{out_path}\n\nControlla i permessi della cartella.\n\n{e}", None)
        return

    class _SegmentCapture:
        encoding = "utf-8"

        def __init__(self):
            self._buf = ""

        def write(self, s):
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._emit(line)
            return len(s)

        def _emit(self, line):
            line = line.strip()
            if not line.startswith("["):
                return
            try:
                inside, text = line[1:].split("]", 1)
                t1, t2 = inside.split(" --> ")
                s, e = _ts_to_sec(t1), _ts_to_sec(t2)
                line = f"[{int(s // 60):02d}:{int(s % 60):02d} - {int(e // 60):02d}:{int(e % 60):02d}] {text.strip()}"
            except Exception:
                pass
            # salvataggio progressivo: scrivi e svuota subito su disco
            try:
                out_file.write(line + "\n")
                out_file.flush()
            except Exception:
                pass
            on_segment(line)

        def flush(self):
            pass

    transcribe_kwargs = dict(language=None, fp16=False, verbose=True)
    if resume_from > 0:
        transcribe_kwargs["clip_timestamps"] = [float(resume_from)]

    real_stdout = sys.stdout
    orig_tqdm = _wt.tqdm
    sys.stdout = _SegmentCapture()
    _wt.tqdm = _FakeTqdmModule
    try:
        result = whisper.transcribe(model, audio_path, **transcribe_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            on_done(False, "Memoria insufficiente!\n\nProva col modello Medium che è più leggero.", None)
        else:
            on_done(False, f"Errore durante la sbobinatura:\n{e}", None)
        return
    except Exception as e:
        on_done(False, f"Errore durante la sbobinatura:\n{e}", None)
        return
    finally:
        sys.stdout = real_stdout
        _wt.tqdm = orig_tqdm
        try:
            out_file.close()
        except Exception:
            pass

    # Riconoscimento dei parlanti (opzionale): pyannote analizza tutto l'audio,
    # poi riscriviamo il file con le etichette e ricarichiamo la finestra.
    if diarize_on:
        try:
            turns = diarize(audio_path, num_speakers=num_speakers, on_status=on_status)
            segs = assign_speakers(result["segments"], turns)
            label_map = speaker_label_map(segs)
            with open(out_path, "w", encoding="utf-8") as f:
                for seg in segs:
                    f.write(format_line(seg, label_map) + "\n")
            on_segment("__RELOAD__")
        except Exception as e:
            on_status(f"Voci non riconosciute: {e}")

    # Il file è già stato scritto in modo incrementale durante la trascrizione.
    elapsed = time.time() - start_time
    detected = result.get("language", "?")
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    on_progress("Completato!", 100)
    on_done(True, out_path, f"{detected} — tempo: {mins}m {secs}s")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sbobinator — Comando Trascrizioni")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        self.root.configure(bg=CARA_BLU)

        # Stile Carabinieri per i widget ttk (barra di avanzamento)
        style = ttk.Style()
        try:
            style.theme_use("default")
        except tk.TclError:
            pass
        style.configure("Cara.Horizontal.TProgressbar",
                        troughcolor=CARA_BLU_SCURO, background=CARA_ROSSO,
                        bordercolor=CARA_ORO, lightcolor=CARA_ROSSO, darkcolor=CARA_ROSSO)

        frame = tk.Frame(root, padx=20, pady=15, bg=CARA_BLU)
        frame.pack(fill="both", expand=True)

        # Banda tricolore dei gradi (oro/rosso) in alto
        banda = tk.Frame(frame, bg=CARA_BLU)
        banda.pack(fill="x", pady=(0, 8))
        tk.Frame(banda, bg=CARA_ROSSO, height=4).pack(fill="x")

        tk.Label(frame, text="🔥  S B O B I N A T O R  🔥", font=("Arial", 18, "bold"),
                 fg=CARA_ORO, bg=CARA_BLU).pack(pady=(0, 2))
        tk.Label(frame, text="Trascrittore audio locale — per il Comando locale",
                 font=("Arial", 9), fg=CARA_TXT, bg=CARA_BLU).pack(pady=(0, 1))
        tk.Label(frame, text="⚜  Nei secoli fedele  ⚜", font=("Arial", 9, "italic"),
                 fg=CARA_ROSSO, bg=CARA_BLU).pack(pady=(0, 6))

        info = get_device_info()
        if info["device"] == "cuda":
            vram = f" ({info['vram_gb']} GB)" if info["vram_gb"] else ""
            device_text = f"⚡ GPU: {info['name']}{vram}"
            device_color = CARA_ORO
        else:
            device_text = "💻 CPU (nessuna GPU CUDA rilevata)"
            device_color = COL_MUTED
        tk.Label(frame, text=device_text, font=("Arial", 9),
                 fg=device_color, bg=CARA_BLU).pack(pady=(0, 10))

        self.model_var = tk.StringVar(value="medium")

        self.model_status = tk.Label(frame, text="", font=("Arial", 9), bg=CARA_BLU)
        self.model_status.pack(pady=(0, 5))
        self.check_models()

        row = tk.Frame(frame, bg=CARA_BLU)
        row.pack(fill="x", pady=(0, 10))
        tk.Label(row, text="Modello:", font=("Arial", 10), fg=CARA_TXT, bg=CARA_BLU).pack(side="left")
        for name, info in MODEL_INFO.items():
            tk.Radiobutton(row, text=info["label"], variable=self.model_var, value=name,
                           font=("Arial", 9), command=self.check_models,
                           bg=CARA_BLU, fg=CARA_TXT, selectcolor=CARA_BLU_SCURO,
                           activebackground=CARA_BLU, activeforeground=CARA_ORO,
                           highlightthickness=0).pack(side="left", padx=(10, 0))

        diar_row = tk.Frame(frame, bg=CARA_BLU)
        diar_row.pack(fill="x", pady=(0, 10))
        self.diar_var = tk.BooleanVar(value=False)
        tk.Checkbutton(diar_row, text="Riconosci chi parla (più lento)",
                       variable=self.diar_var, font=("Arial", 9),
                       bg=CARA_BLU, fg=CARA_TXT, selectcolor=CARA_BLU_SCURO,
                       activebackground=CARA_BLU, activeforeground=CARA_ORO,
                       highlightthickness=0).pack(side="left")
        tk.Label(diar_row, text="Voci attese:", font=("Arial", 9),
                 fg=CARA_TXT, bg=CARA_BLU).pack(side="left", padx=(12, 4))
        self.speakers_var = tk.StringVar(value="auto")
        ttk.Combobox(diar_row, textvariable=self.speakers_var, width=5, state="readonly",
                     values=["auto", "2", "3", "4", "5", "6"]).pack(side="left")
        self.rename_btn = tk.Button(diar_row, text="Rinomina voci", command=self.rename_speakers,
                                    font=("Arial", 9), bg=CARA_BLU_SCURO, fg=CARA_TXT,
                                    activebackground=CARA_ORO, activeforeground=CARA_BLU,
                                    relief="flat", state="disabled")
        self.rename_btn.pack(side="right")

        # --- Profilo qualità + opzioni avanzate pulizia audio ---
        profile_row = tk.Frame(frame, bg=CARA_BLU)
        profile_row.pack(fill="x", pady=(0, 4))
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
                           command=self._on_profile_change).pack(side="left", padx=(6, 0))

        adv_row = tk.Frame(frame, bg=CARA_BLU)
        adv_row.pack(fill="x", pady=(0, 8))
        self.normalize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(adv_row, text="Normalizza volume (opzionale, per audio con livelli variabili)",
                       variable=self.normalize_var, font=("Arial", 9),
                       bg=CARA_BLU, fg=CARA_TXT, selectcolor=CARA_BLU_SCURO,
                       activebackground=CARA_BLU, activeforeground=CARA_ORO,
                       highlightthickness=0).pack(side="left")

        btn_row = tk.Frame(frame, bg=CARA_BLU)
        btn_row.pack(fill="x", pady=(0, 10))

        self.btn = tk.Button(btn_row, text="Seleziona file audio", command=self.pick_file,
                             height=2, font=("Arial", 11, "bold"),
                             bg=CARA_ROSSO, fg="white", activebackground=CARA_ORO,
                             activeforeground=CARA_BLU, relief="flat", cursor="hand2")
        self.btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.dl_btn = tk.Button(btn_row, text="Scarica\nmodello", command=self.download_model_ui,
                                height=2, font=("Arial", 9), width=10,
                                bg=CARA_BLU_SCURO, fg=CARA_TXT, activebackground=CARA_ORO,
                                activeforeground=CARA_BLU, relief="flat", cursor="hand2")
        self.dl_btn.pack(side="right")

        self.status = tk.Label(frame, text="Pronto", font=("Arial", 10), fg=COL_MUTED, bg=CARA_BLU)
        self.status.pack(pady=(0, 5))

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100,
                                        style="Cara.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(0, 5))

        self.pct_label = tk.Label(frame, text="", font=("Arial", 9), fg=COL_MUTED, bg=CARA_BLU)
        self.pct_label.pack(pady=(0, 10))

        tk.Label(frame, text="Trascrizione live:", font=("Arial", 9, "bold"), anchor="w",
                 fg=CARA_TXT, bg=CARA_BLU).pack(fill="x")
        text_frame = tk.Frame(frame, bg=CARA_BLU)
        text_frame.pack(fill="both", expand=True, pady=(3, 0))

        self.transcript = tk.Text(text_frame, height=10, font=("Consolas", 9), wrap="word",
                                  state="disabled", bg=CARA_BLU_SCURO, fg=CARA_TXT,
                                  insertbackground=CARA_TXT, relief="flat",
                                  highlightthickness=1, highlightbackground=CARA_ORO)
        scrollbar = ttk.Scrollbar(text_frame, command=self.transcript.yview)
        self.transcript.config(yscrollcommand=scrollbar.set)
        self.transcript.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def check_models(self):
        model = self.model_var.get()
        if model_exists(model):
            self.model_status.config(text=f"✓ Modello {model} trovato", fg=COL_OK)
        else:
            self.model_status.config(
                text=f"✗ Modello {model} non trovato — verrà scaricato al primo uso (serve internet)",
                fg=COL_WARN)

    def download_model_ui(self):
        model_name = self.model_var.get()
        if model_exists(model_name):
            messagebox.showinfo("Sbobinator", f"Il modello {model_name} è già scaricato!")
            return

        self.btn.config(state="disabled")
        self.dl_btn.config(state="disabled")

        def on_progress(msg, pct):
            def update():
                self.pct_label.config(text=msg)
                if pct >= 0:
                    self.progress["mode"] = "determinate"
                    self.progress["value"] = pct
                else:
                    self.progress["mode"] = "indeterminate"
                    self.progress.start(10)
            self.root.after(0, update)

        def do_download():
            self.root.after(0, lambda: self.status.config(text=f"Download modello {model_name}...", fg=COL_INFO))
            ok, err = download_model(model_name, on_progress)
            def finish():
                try:
                    self.progress.stop()
                except Exception:
                    pass
                self.progress["mode"] = "determinate"
                self.btn.config(state="normal")
                self.dl_btn.config(state="normal")
                self.check_models()
                if ok:
                    self.progress["value"] = 100
                    self.status.config(text=f"Modello {model_name} scaricato!", fg=COL_OK)
                    messagebox.showinfo("Sbobinator", f"Modello {model_name} scaricato nella cartella models/.\n\nOra puoi copiare tutta la cartella su chiavetta per l'uso offline.")
                else:
                    self.progress["value"] = 0
                    self.status.config(text="Errore download", fg=COL_ERR)
                    messagebox.showerror("Errore", f"Download fallito:\n{err}")
            self.root.after(0, finish)

        t = threading.Thread(target=do_download, daemon=True)
        t.start()

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Seleziona file audio",
            filetypes=[
                ("Audio", "*.mp3 *.wav *.m4a *.ogg *.flac *.wma *.aac *.mp4 *.webm *.opus"),
                ("Tutti i file", "*.*"),
            ],
        )
        if not path:
            return

        resume_from = 0.0
        out_path = os.path.splitext(path)[0] + ".txt"
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            last = last_transcribed_time(out_path)
            if last > 0:
                ans = messagebox.askyesnocancel(
                    "Sbobinator",
                    f"Esiste già una trascrizione per questo audio, arrivata a "
                    f"{int(last // 60):02d}:{int(last % 60):02d}.\n\n"
                    "Sì = riprendi da lì\n"
                    "No = ricomincia da capo\n"
                    "Annulla = lascia stare",
                )
                if ans is None:
                    return
                resume_from = last if ans else 0.0

        self.start_transcription(path, resume_from)

    def add_segment(self, line):
        def update():
            if line == "__RELOAD__":
                # ricarica il file (ora con le etichette dei parlanti)
                self.transcript.config(state="normal")
                self.transcript.delete("1.0", "end")
                try:
                    with open(self._current_out, "r", encoding="utf-8") as f:
                        self.transcript.insert("end", f.read())
                except Exception:
                    pass
                self.transcript.see("end")
                self.transcript.config(state="disabled")
                return
            self.transcript.config(state="normal")
            self.transcript.insert("end", line + "\n")
            self.transcript.see("end")
            self.transcript.config(state="disabled")
        self.root.after(0, update)

    def start_transcription(self, path, resume_from=0.0):
        self.btn.config(state="disabled")
        self.transcript.config(state="normal")
        self.transcript.delete("1.0", "end")
        if resume_from > 0:
            # mostra ciò che era già stato trascritto, poi si accoda il resto
            try:
                with open(os.path.splitext(path)[0] + ".txt", "r", encoding="utf-8") as f:
                    self.transcript.insert("end", f.read())
            except Exception:
                pass
            self.transcript.see("end")
        self.transcript.config(state="disabled")
        self.progress["value"] = 0

        model_name = self.model_var.get()
        self._current_out = os.path.splitext(path)[0] + ".txt"
        diarize_on = self.diar_var.get()
        sp = self.speakers_var.get()
        num_speakers = None if sp == "auto" else int(sp)
        self.rename_btn.config(state="disabled")

        def on_status(msg):
            self.root.after(0, lambda: self.status.config(text=msg, fg=COL_INFO))

        def on_progress(msg, pct):
            def update():
                self.pct_label.config(text=msg)
                if pct >= 0:
                    self.progress["mode"] = "determinate"
                    self.progress["value"] = pct
                else:
                    self.progress["mode"] = "indeterminate"
                    self.progress.start(10)
            self.root.after(0, update)

        def on_segment(line):
            self.add_segment(line)

        def on_done(success, result, info):
            def update():
                try:
                    self.progress.stop()
                except Exception:
                    pass
                self.progress["mode"] = "determinate"
                self.btn.config(state="normal")
                if success:
                    self.progress["value"] = 100
                    self.status.config(text=f"Fatto! {info}", fg=COL_OK)
                    self.pct_label.config(text="100%")
                    if "Interlocutore" in self.transcript.get("1.0", "end"):
                        self.rename_btn.config(state="normal")
                    messagebox.showinfo("Sbobinator", f"Trascrizione salvata in:\n{result}")
                else:
                    self.progress["value"] = 0
                    self.status.config(text="Errore", fg=COL_ERR)
                    self.pct_label.config(text="")
                    messagebox.showerror("Errore", result)
            self.root.after(0, update)

        callbacks = (on_status, on_progress, on_segment, on_done)
        normalize_on = self.normalize_var.get()
        t = threading.Thread(
            target=transcribe,
            args=(path, model_name, callbacks, resume_from,
                  diarize_on, num_speakers, normalize_on),
            daemon=True,
        )
        t.start()

    def _on_profile_change(self):
        # Con l'app snellita (solo ASR raw + opzionale normalizzazione)
        # i profili si limitano a impostare normalize on/off.
        # Whisper/Voxtral sono stati addestrati su audio raw — non serve
        # ripulire ulteriormente, peggiorerebbe i risultati.
        p = self.profile_var.get()
        if p == "standard":
            self.normalize_var.set(False)
        elif p == "intercettazione":
            self.normalize_var.set(True)  # tipico forense: livelli variabili
        elif p == "max":
            self.normalize_var.set(True)
        # advanced: lascia la scelta com'è

    def rename_speakers(self):
        import re
        text = self.transcript.get("1.0", "end")
        found = sorted(set(re.findall(r"Interlocutore \d+", text)),
                       key=lambda s: int(s.split()[1]))
        if not found:
            return
        win = tk.Toplevel(self.root)
        win.title("Rinomina voci")
        win.configure(bg=CARA_BLU)
        entries = {}
        for i, name in enumerate(found):
            tk.Label(win, text=name + "  →", bg=CARA_BLU, fg=CARA_TXT,
                     font=("Arial", 10)).grid(row=i, column=0, padx=8, pady=4, sticky="e")
            entry = tk.Entry(win, width=24)
            entry.grid(row=i, column=1, padx=8, pady=4)
            entries[name] = entry

        def apply():
            mapping = {old: e.get().strip() for old, e in entries.items() if e.get().strip()}
            if mapping:
                try:
                    with open(self._current_out, "r", encoding="utf-8") as f:
                        content = f.read()
                    for old, new in mapping.items():
                        content = content.replace(old + ":", new + ":")
                    with open(self._current_out, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    pass
                self.transcript.config(state="normal")
                box = self.transcript.get("1.0", "end")
                for old, new in mapping.items():
                    box = box.replace(old + ":", new + ":")
                self.transcript.delete("1.0", "end")
                self.transcript.insert("end", box.rstrip("\n") + "\n")
                self.transcript.config(state="disabled")
            win.destroy()

        tk.Button(win, text="Applica", command=apply, bg=CARA_ROSSO, fg="white",
                  relief="flat").grid(row=len(found), column=0, columnspan=2, pady=10)


def main():
    # Modalità diagnostica per validare l'exe frozen sul PC target.
    # Uso: Sbobinator.exe --selftest <audio>
    # Scrive un .selftest.log accanto all'audio, exit 0 se tutto OK.
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        if len(sys.argv) < 3:
            sys.stderr.write("uso: Sbobinator.exe --selftest <audio>\n") if sys.stderr else None
            sys.exit(2)
        audio = sys.argv[2]
        log_path = os.path.splitext(audio)[0] + ".selftest.log"
        with open(log_path, "w", encoding="utf-8") as log:
            def w(msg):
                log.write(msg + "\n"); log.flush()
            try:
                w(f"device: {get_device_info()}")
                w(f"audio: {audio}")
                w("checking ffmpeg...")
                w(f"  found: {check_ffmpeg()}")
                w("loading audio...")
                a = load_audio_array(audio)
                w(f"  samples: {len(a)}")
                # niente denoise nello stack v3 snellito — Whisper/Voxtral raw
                w("(skip denoise — app snellita)")
                w("diarizing...")
                turns = diarize(audio)
                w(f"  turns: {len(turns)}")
                w("SELFTEST OK")
                sys.exit(0)
            except Exception as e:
                import traceback
                w(f"SELFTEST FAIL: {e}")
                w(traceback.format_exc())
                sys.exit(1)

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if os.path.isfile(audio_path):
            if not check_ffmpeg():
                print("ERRORE: ffmpeg non trovato. Mettilo nella stessa cartella.")
                sys.exit(1)
            print(f"Sbobinatura: {audio_path}")
            model_name = "medium"
            if not model_exists(model_name):
                print("Modello non trovato, download...")
                ok, err = download_model(model_name, lambda msg, _pct: print(f"\r{msg}", end="", flush=True))
                print()
                if not ok:
                    print(f"Errore download: {err}")
                    sys.exit(1)
            print("Caricamento modello...")
            import whisper
            model = whisper.load_model(model_name, download_root=get_models_dir(), device=get_device())
            print("Sbobinatura in corso...")
            result = model.transcribe(audio_path, language=None, fp16=False)
            out_path = os.path.splitext(audio_path)[0] + ".txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for seg in result["segments"]:
                    start = seg["start"]
                    end = seg["end"]
                    m1, s1 = int(start // 60), int(start % 60)
                    m2, s2 = int(end // 60), int(end % 60)
                    text = seg["text"].strip()
                    line = f"[{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}] {text}"
                    f.write(line + "\n")
                    print(line)
            detected = result.get("language", "?")
            print(f"\nFatto! Lingua: {detected}")
            print(f"Salvato in: {out_path}")
            return

    root = tk.Tk()
    App(root)
    # Chiusura pulita: alla X termina tutto, e os._exit forza la fine completa
    # del processo (e del processo figlio creato da PyInstaller "onefile"),
    # così non restano processi orfani/ghost anche se una trascrizione è in
    # corso al momento della chiusura.
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    os._exit(0)


if __name__ == "__main__":
    main()
