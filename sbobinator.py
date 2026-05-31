import sys
import os
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import warnings
import platform
import traceback

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

MODEL_NAME = "large"
MODEL_FILE = "large-v3.pt"
MODEL_URL = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
MODEL_SIZE_GB = 2.9

APP_NAME = "verbaLIA"
APP_VERSION = "3.1"
APP_SUBTITLE = "Trascrizione audio locale"

# Palette dark moderna
BG = "#0f1419"          # sfondo principale
SURFACE = "#161b22"     # card / pannelli
SURFACE_2 = "#1f242e"   # input / hover
BORDER = "#2a3142"
TXT = "#e6e8eb"
TXT_MUTED = "#7d8590"
ACCENT = "#4a9eff"      # blu primario
ACCENT_HOVER = "#5fb0ff"
ACCENT_FG = "#0f1419"
OK = "#3fb950"
WARN = "#d29922"
ERR = "#f85149"

FONT = "Segoe UI"


def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ---------- Log di esecuzione ----------
# File di log accanto all'eseguibile: registra ogni passo e i traceback completi
# degli errori. Pensato per essere inviato allo sviluppatore per il debug.
LOG_MAX_BYTES = 2_000_000   # ~2 MB poi ruota (.1) per non crescere all'infinito


def get_log_path():
    return os.path.join(get_base_path(), "verbalia_debug.log")


def _log_write(text):
    try:
        path = get_log_path()
        if os.path.isfile(path) and os.path.getsize(path) > LOG_MAX_BYTES:
            try:
                os.replace(path, path + ".1")
            except Exception:
                pass
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def log_event(msg, level="INFO"):
    """Una riga nel log di esecuzione, con timestamp e livello."""
    _log_write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n")


def log_exception(context, exc):
    """Logga un errore con il traceback completo (per il debug)."""
    log_event(f"{context}: {exc}", level="ERROR")
    _log_write(traceback.format_exc() + "\n")


def log_session_header(config=None):
    """Intestazione di sessione con hardware e configurazione scelta."""
    _log_write("\n" + "=" * 66 + "\n")
    log_event(f"verbaLIA {APP_VERSION} — nuova sessione")
    try:
        log_event(f"OS: {platform.platform()}")
        sysinfo = get_system_info()
        log_event(f"CPU: {sysinfo['cores']} core  ·  RAM: {sysinfo['ram_gb']} GB")
        info = get_device_info()
        log_event(f"GPU: {info['name']}  ·  VRAM: {info.get('vram_gb')} GB")
    except Exception as e:
        log_event(f"rilevamento hardware fallito: {e}", level="WARN")
    if config:
        log_event(f"Config runtime: device={config['device']} fp16={config['fp16']} "
                  f"mode={config['mode']} — {config['reason']}")


def get_models_dir():
    d = os.path.join(get_base_path(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_exists():
    return os.path.isfile(os.path.join(get_models_dir(), MODEL_FILE))


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


# Soglie di policy per la scelta automatica del runtime (VRAM in GB). NON sono
# parametri hardware cablati: sono soglie regolabili usate solo come default
# quando l'utente lascia "auto". large-v3 in fp16 pesa ~3,1 GB di pesi + overhead.
VRAM_MIN_GB = 4.0          # sotto: large-v3 non entra in GPU → CPU
VRAM_COMFORTABLE_GB = 4.8  # sopra: GPU "comoda"; tra MIN e questa: "memoria ridotta"


def get_system_info():
    """RAM totale (GB) e core logici, per la diagnostica. Best-effort."""
    cores = os.cpu_count() or 0
    ram_gb = None
    try:
        if sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            ram_gb = round(ms.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        ram_gb = None
    return {"cores": cores, "ram_gb": ram_gb}


def decide_runtime_config(override="auto"):
    """Sceglie device, precisione e thread in base all'hardware rilevato.

    override: "auto" (rileva), "gpu" (forza GPU se presente), "cpu" (forza CPU).
    Funzione pura: nessun effetto collaterale, ritorna un dict descrittivo che
    l'app usa per configurarsi e per mostrare cosa sta facendo. Niente valori
    cablati: tutto deriva dall'hardware letto a runtime.
    """
    info = get_device_info()          # {device, name, vram_gb}
    sysinfo = get_system_info()
    has_cuda = info["device"] == "cuda"
    vram = info.get("vram_gb")
    name = info.get("name")

    device, mode, reason = "cpu", "normale", ""

    if override == "cpu":
        device, reason = "cpu", "Forzato su CPU dall'utente."
    elif override == "gpu":
        if has_cuda:
            device, mode, reason = "cuda", "forzata", "Forzato su GPU dall'utente."
        else:
            device, reason = "cpu", "GPU forzata ma nessuna NVIDIA rilevata: uso CPU."
    else:  # auto
        if has_cuda and vram is not None and vram >= VRAM_MIN_GB:
            device = "cuda"
            if vram >= VRAM_COMFORTABLE_GB:
                mode = "comoda"
                reason = f"{name} ({vram} GB VRAM): GPU + fp16."
            else:
                mode = "memoria ridotta"
                reason = (f"{name} ({vram} GB VRAM, al limite per large-v3): provo "
                          "GPU in fp16, ripiego su CPU se la memoria finisce.")
        elif has_cuda:
            device = "cpu"
            reason = (f"{name} ({vram} GB VRAM): troppa poca memoria per large-v3, "
                      "uso CPU.")
        else:
            device, reason = "cpu", "Nessuna GPU NVIDIA: uso CPU."

    return {
        "device": device,
        "fp16": device == "cuda",
        "threads": sysinfo["cores"] if device == "cpu" else None,
        "mode": mode,
        "reason": reason,
        "vram_gb": vram,
        "gpu_name": name if has_cuda else None,
        "ram_gb": sysinfo["ram_gb"],
        "cores": sysinfo["cores"],
        "override": override,
    }


def gather_component_status():
    """Stato dei componenti per il pannello diagnostica.

    Funzione pura (nessuna UI) così è testabile a parte. Ritorna una lista di
    dict con: key, label, what (a cosa serve), ok (bool), detail, neutral
    (assenza non è un errore, es. GPU su un PC senza NVIDIA).
    """
    items = []
    items.append({
        "key": "ffmpeg",
        "label": "ffmpeg",
        "what": "Converte e legge i file audio/video. Necessario.",
        "ok": check_ffmpeg(),
        "detail": "",
        "neutral": False,
    })

    mok = model_exists()
    items.append({
        "key": "model",
        "label": "Modello Whisper large-v3",
        "what": "Il motore della trascrizione. Obbligatorio.",
        "ok": mok,
        "detail": "Pronto" if mok else f"Assente — scaricalo (~{MODEL_SIZE_GB:.1f} GB)",
        "neutral": False,
    })

    info = get_device_info()
    gpu_ok = info["device"] == "cuda"
    if gpu_ok:
        vram = f", {info['vram_gb']} GB VRAM" if info.get("vram_gb") else ""
        detail = f"{info['name']}{vram}"
    else:
        detail = "Nessuna GPU NVIDIA rilevata su questo PC → uso la CPU"
    items.append({
        "key": "gpu",
        "label": "Accelerazione GPU (NVIDIA / CUDA)",
        "what": "Velocizza molto la trascrizione sui PC con scheda NVIDIA. "
                "Opzionale: senza, l'app usa la CPU. Le librerie CUDA sono già "
                "incluse nell'app, non serve scaricare nulla.",
        "ok": gpu_ok,
        "detail": detail,
        "neutral": not gpu_ok,
    })
    return items


def download_model(progress_callback):
    import urllib.request

    dest = os.path.join(get_models_dir(), MODEL_FILE)
    dest_tmp = dest + ".downloading"

    try:
        req = urllib.request.urlopen(MODEL_URL)
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


def diarize(audio_path, num_speakers=None, on_status=None, device=None):
    """Diarization offline. Ritorna [(start, end, label)].

    Usa pyannote 4.x. L'audio viene caricato in memoria col ffmpeg di whisper
    (evita torchcodec). I modelli sono caricati dalla cache locale (offline);
    nella build congelata stanno in 'hf_models/' dentro il bundle.

    device: "cuda"/"cpu" coerente con la scelta runtime; se None usa get_device().
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
    pipe.to(torch.device(device or get_device()))

    audio = whisper.audio.load_audio(audio_path)  # float32 mono @ 16 kHz
    waveform = torch.from_numpy(audio).unsqueeze(0)
    call_kwargs = {}
    if num_speakers:
        call_kwargs["num_speakers"] = int(num_speakers)
    dia = pipe({"waveform": waveform, "sample_rate": 16000}, **call_kwargs)
    ann = dia.speaker_diarization
    return [(t.start, t.end, label) for t, _, label in ann.itertracks(yield_label=True)]


def transcribe(audio_path, ui_callbacks, resume_from=0.0,
               diarize_on=False, num_speakers=None,
               normalize_on=False, preloaded_model=None,
               runtime_config=None, on_fallback=None):
    on_status, on_progress, on_segment, on_done = ui_callbacks

    if runtime_config is None:
        runtime_config = decide_runtime_config("auto")

    # "Parlante": ogni messaggio di stato mostrato all'utente va anche nel log
    # di esecuzione, così i passi compaiono sia nella UI sia nel file di debug.
    _ui_on_status = on_status

    def on_status(msg, level="INFO"):
        try:
            _ui_on_status(msg)
        except Exception:
            pass
        log_event(msg, level=level)

    log_event(f"--- Trascrizione: {os.path.basename(audio_path)}  "
              f"(diar={diarize_on}, norm={normalize_on}, resume={resume_from:.0f}s, "
              f"device={runtime_config['device']}, fp16={runtime_config['fp16']})")

    if normalize_on:
        try:
            audio_path, _ = normalize_volume(audio_path, on_status=on_status)
        except Exception as e:
            on_status(f"Normalizzazione fallita ({e}); proseguo con l'originale.")

    if not check_ffmpeg():
        on_done(False, "ffmpeg non trovato.\n\nMetti ffmpeg.exe nella stessa cartella dell'eseguibile.", None)
        return

    import whisper
    if preloaded_model is not None:
        model = preloaded_model
    else:
        if not model_exists():
            on_status("Modello non trovato. Download in corso...")
            ok, err = download_model(lambda msg, pct: on_progress(msg, pct))
            if not ok:
                on_done(False, f"Errore download modello:\n{err}\n\nSe il PC non ha internet, copia large-v3.pt in models/.", None)
                return
        dev = runtime_config["device"]
        on_status(f"Caricamento modello su {dev.upper()}"
                  f"{' (fp16)' if runtime_config['fp16'] else ''}...")
        on_progress("Caricamento modello...", -1)
        try:
            model = whisper.load_model(MODEL_NAME, download_root=get_models_dir(),
                                       device=dev)
        except Exception as e:
            on_done(False, f"Errore caricamento modello:\n{e}\n\nIl file potrebbe essere corrotto. Cancella la cartella models/ e riscarica.", None)
            log_exception("Caricamento modello fallito", e)
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
        out_file.seek(0, os.SEEK_END)
        out_file_start = out_file.tell()   # punto da cui ripartire se serve azzerare
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

    transcribe_kwargs = dict(language=None, fp16=runtime_config["fp16"], verbose=True)
    if resume_from > 0:
        transcribe_kwargs["clip_timestamps"] = [float(resume_from)]

    real_stdout = sys.stdout
    orig_tqdm = _wt.tqdm
    sys.stdout = _SegmentCapture()
    _wt.tqdm = _FakeTqdmModule
    try:
        try:
            result = whisper.transcribe(model, audio_path, **transcribe_kwargs)
        except RuntimeError as e:
            # Ripiego automatico GPU → CPU se la VRAM è insufficiente per large-v3
            # (caso tipico: schede con poca memoria come la T1000 4 GB).
            if "out of memory" in str(e).lower() and runtime_config["device"] == "cuda":
                log_exception("Out-of-memory su GPU, ripiego automatico su CPU", e)
                on_status("VRAM insufficiente per la GPU: passo alla CPU e riprovo...",
                          level="WARN")
                try:
                    import torch
                    del model
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                model = whisper.load_model(MODEL_NAME,
                                           download_root=get_models_dir(), device="cpu")
                runtime_config = dict(runtime_config, device="cpu", fp16=False,
                                      mode="ripiego CPU")
                transcribe_kwargs["fp16"] = False
                # azzera l'eventuale output parziale del tentativo GPU fallito
                # (preservando il prefisso del resume) per non duplicare segmenti
                try:
                    out_file.seek(out_file_start)
                    out_file.truncate()
                except Exception:
                    pass
                if on_fallback:
                    try:
                        on_fallback(model, runtime_config)
                    except Exception:
                        pass
                result = whisper.transcribe(model, audio_path, **transcribe_kwargs)
            else:
                raise
    except RuntimeError as e:
        on_done(False, f"Errore durante la sbobinatura:\n{e}", None)
        log_exception(f"Trascrizione fallita: {os.path.basename(audio_path)}", e)
        return
    except Exception as e:
        on_done(False, f"Errore durante la sbobinatura:\n{e}", None)
        log_exception(f"Trascrizione fallita: {os.path.basename(audio_path)}", e)
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
            turns = diarize(audio_path, num_speakers=num_speakers,
                            on_status=on_status, device=runtime_config["device"])
            segs = assign_speakers(result["segments"], turns)
            label_map = speaker_label_map(segs)
            with open(out_path, "w", encoding="utf-8") as f:
                for seg in segs:
                    f.write(format_line(seg, label_map) + "\n")
            on_segment("__RELOAD__")
        except Exception as e:
            on_status(f"Voci non riconosciute: {e}", level="WARN")
            log_exception("Diarizzazione fallita", e)

    # Il file è già stato scritto in modo incrementale durante la trascrizione.
    elapsed = time.time() - start_time
    detected = result.get("language", "?")
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    on_progress("Completato!", 100)
    on_done(True, out_path, f"{detected} — tempo: {mins}m {secs}s")


AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma",
              ".aac", ".mp4", ".webm", ".opus")


def fmt_duration(seconds):
    if seconds is None:
        return "—"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class FlatButton(tk.Button):
    """Button flat con hover, padding generoso. Primario/secondario via stile."""

    def __init__(self, master, text, command, kind="primary", **kw):
        if kind == "primary":
            bg, fg, hover = ACCENT, ACCENT_FG, ACCENT_HOVER
        elif kind == "danger":
            bg, fg, hover = SURFACE_2, ERR, "#2a2027"
        else:  # secondary
            bg, fg, hover = SURFACE_2, TXT, BORDER
        super().__init__(
            master, text=text, command=command,
            bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
            font=(FONT, 10), relief="flat", borderwidth=0,
            padx=18, pady=9, cursor="hand2", **kw,
        )
        self._bg, self._hover = bg, hover
        self.bind("<Enter>", lambda _e: self.configure(bg=self._hover))
        self.bind("<Leave>", lambda _e: self.configure(bg=self._bg))

    def set_kind(self, kind):
        if kind == "primary":
            self._bg, self._hover = ACCENT, ACCENT_HOVER
        elif kind == "danger":
            self._bg, self._hover = SURFACE_2, "#2a2027"
        else:
            self._bg, self._hover = SURFACE_2, BORDER
        self.configure(bg=self._bg)


# Stati item della coda
ST_PENDING = "pending"
ST_RUNNING = "running"
ST_DONE = "done"
ST_SKIPPED = "skipped"
ST_ERROR = "error"

STATE_BADGE = {
    ST_PENDING: ("In attesa", TXT_MUTED),
    ST_RUNNING: ("In corso", ACCENT),
    ST_DONE: ("Completato", OK),
    ST_SKIPPED: ("Saltato", TXT_MUTED),
    ST_ERROR: ("Errore", ERR),
}


class QueueItem:
    __slots__ = ("path", "duration", "state", "resume_from", "error")

    def __init__(self, path):
        self.path = path
        self.duration = get_audio_duration(path)
        self.state = ST_PENDING
        self.resume_from = 0.0
        self.error = None


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("820x780")
        self.root.minsize(720, 640)
        self.root.configure(bg=BG)

        self.items = []                # lista QueueItem
        self.row_to_index = {}         # treeview iid -> idx
        self.current_idx = None        # indice item in elaborazione
        self.queue_running = False
        self.stop_requested = False
        self.whisper_model = None      # caricato una volta sola, riusato per tutta la coda
        self._model_device = None      # device su cui è caricato il modello in cache
        self.device_override = tk.StringVar(value="auto")  # auto | gpu | cpu
        self._runtime_config = None    # config scelta a inizio coda

        self._init_ttk_style()
        self._build_ui()
        self._refresh_model_status()

    # ---------- stile ttk ----------

    def _init_ttk_style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass

        s.configure("Verb.Horizontal.TProgressbar",
                    troughcolor=SURFACE_2, background=ACCENT,
                    bordercolor=SURFACE, lightcolor=ACCENT, darkcolor=ACCENT,
                    thickness=8)

        s.configure("Verb.Treeview",
                    background=SURFACE, fieldbackground=SURFACE,
                    foreground=TXT, bordercolor=BORDER, borderwidth=0,
                    rowheight=28, font=(FONT, 10))
        s.configure("Verb.Treeview.Heading",
                    background=SURFACE_2, foreground=TXT_MUTED,
                    font=(FONT, 9, "bold"), relief="flat", borderwidth=0)
        s.map("Verb.Treeview.Heading",
              background=[("active", SURFACE_2)])
        s.map("Verb.Treeview",
              background=[("selected", "#2a3f5f")],
              foreground=[("selected", TXT)])

        s.configure("Verb.TCombobox",
                    fieldbackground=SURFACE_2, background=SURFACE_2,
                    foreground=TXT, borderwidth=0, arrowcolor=TXT_MUTED)
        s.map("Verb.TCombobox",
              fieldbackground=[("readonly", SURFACE_2)],
              foreground=[("readonly", TXT)])

        s.configure("Verb.Vertical.TScrollbar",
                    background=SURFACE_2, troughcolor=BG,
                    bordercolor=BG, arrowcolor=TXT_MUTED, borderwidth=0)

    # ---------- costruzione UI ----------

    def _build_ui(self):
        root_pad = tk.Frame(self.root, bg=BG, padx=24, pady=20)
        root_pad.pack(fill="both", expand=True)

        self._build_header(root_pad)
        self._build_options(root_pad)
        self._build_queue(root_pad)
        self._build_action_bar(root_pad)
        self._build_status(root_pad)
        self._build_preview(root_pad)

    def _build_header(self, parent):
        head = tk.Frame(parent, bg=BG)
        head.pack(fill="x", pady=(0, 18))

        left = tk.Frame(head, bg=BG)
        left.pack(side="left")
        tk.Label(left, text=APP_NAME, font=(FONT, 22, "bold"),
                 fg=TXT, bg=BG).pack(anchor="w")
        tk.Label(left, text=APP_SUBTITLE, font=(FONT, 10),
                 fg=TXT_MUTED, bg=BG).pack(anchor="w", pady=(2, 0))

        right = tk.Frame(head, bg=BG)
        right.pack(side="right")
        info = get_device_info()
        if info["device"] == "cuda":
            vram = f"  ·  {info['vram_gb']} GB VRAM" if info["vram_gb"] else ""
            dev_text = f"GPU  ·  {info['name']}{vram}"
            dev_col = OK
        else:
            dev_text = "CPU  ·  nessuna GPU CUDA"
            dev_col = TXT_MUTED
        self.device_lbl = tk.Label(right, text=dev_text, font=(FONT, 9),
                                   fg=dev_col, bg=BG)
        self.device_lbl.pack(anchor="e")
        self.model_lbl = tk.Label(right, text="", font=(FONT, 9),
                                  fg=TXT_MUTED, bg=BG)
        self.model_lbl.pack(anchor="e", pady=(4, 0))
        self.download_btn = FlatButton(right, "Scarica modello",
                                       self.download_model_ui, kind="secondary")
        # configure smaller padding for header context
        self.download_btn.configure(padx=12, pady=6, font=(FONT, 9))

        self.status_btn = FlatButton(right, "Stato componenti",
                                     self.show_components_ui, kind="secondary")
        self.status_btn.configure(padx=12, pady=6, font=(FONT, 9))
        self.status_btn.pack(anchor="e", pady=(6, 0))

    def _build_options(self, parent):
        card = tk.Frame(parent, bg=SURFACE,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x", pady=(0, 14))
        inner = tk.Frame(card, bg=SURFACE, padx=16, pady=12)
        inner.pack(fill="x")

        tk.Label(inner, text="OPZIONI", font=(FONT, 8, "bold"),
                 fg=TXT_MUTED, bg=SURFACE).pack(anchor="w")

        row = tk.Frame(inner, bg=SURFACE)
        row.pack(fill="x", pady=(8, 0))

        self.diar_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, text="Riconosci i parlanti",
                       variable=self.diar_var, font=(FONT, 10),
                       bg=SURFACE, fg=TXT, selectcolor=SURFACE_2,
                       activebackground=SURFACE, activeforeground=TXT,
                       highlightthickness=0, borderwidth=0).pack(side="left")

        tk.Label(row, text="Voci attese", font=(FONT, 9),
                 fg=TXT_MUTED, bg=SURFACE).pack(side="left", padx=(18, 6))
        self.speakers_var = tk.StringVar(value="auto")
        ttk.Combobox(row, textvariable=self.speakers_var, width=6,
                     state="readonly", style="Verb.TCombobox",
                     values=["auto", "2", "3", "4", "5", "6"]).pack(side="left")

        self.normalize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, text="Normalizza volume",
                       variable=self.normalize_var, font=(FONT, 10),
                       bg=SURFACE, fg=TXT, selectcolor=SURFACE_2,
                       activebackground=SURFACE, activeforeground=TXT,
                       highlightthickness=0, borderwidth=0).pack(side="left", padx=(28, 0))

    def _build_queue(self, parent):
        card = tk.Frame(parent, bg=SURFACE,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="both", expand=False, pady=(0, 14))
        inner = tk.Frame(card, bg=SURFACE, padx=16, pady=12)
        inner.pack(fill="both", expand=True)

        header_row = tk.Frame(inner, bg=SURFACE)
        header_row.pack(fill="x")
        tk.Label(header_row, text="CODA DI LAVORO", font=(FONT, 8, "bold"),
                 fg=TXT_MUTED, bg=SURFACE).pack(side="left")
        self.queue_count = tk.Label(header_row, text="0 file",
                                    font=(FONT, 9), fg=TXT_MUTED, bg=SURFACE)
        self.queue_count.pack(side="right")

        tree_wrap = tk.Frame(inner, bg=SURFACE_2,
                             highlightthickness=1, highlightbackground=BORDER)
        tree_wrap.pack(fill="both", expand=True, pady=(8, 10))

        self.tree = ttk.Treeview(
            tree_wrap, style="Verb.Treeview",
            columns=("state", "file", "duration"),
            show="headings", height=8, selectmode="extended",
        )
        self.tree.heading("state", text="Stato", anchor="w")
        self.tree.heading("file", text="File", anchor="w")
        self.tree.heading("duration", text="Durata", anchor="e")
        self.tree.column("state", width=110, minwidth=90, anchor="w", stretch=False)
        self.tree.column("file", width=480, minwidth=200, anchor="w")
        self.tree.column("duration", width=80, minwidth=60, anchor="e", stretch=False)

        self.tree.tag_configure(ST_PENDING, foreground=TXT)
        self.tree.tag_configure(ST_RUNNING, foreground=ACCENT)
        self.tree.tag_configure(ST_DONE, foreground=OK)
        self.tree.tag_configure(ST_SKIPPED, foreground=TXT_MUTED)
        self.tree.tag_configure(ST_ERROR, foreground=ERR)

        vsb = ttk.Scrollbar(tree_wrap, orient="vertical",
                            command=self.tree.yview,
                            style="Verb.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", self._on_queue_double_click)
        vsb.pack(side="right", fill="y")

        tk.Label(inner,
                 text="Suggerimento: doppio click su un file completato per "
                      "riascoltare gli spezzoni e correggere il testo.",
                 font=(FONT, 9), fg=TXT_MUTED, bg=SURFACE,
                 anchor="w", justify="left").pack(fill="x", pady=(0, 8))

        actions = tk.Frame(inner, bg=SURFACE)
        actions.pack(fill="x")
        FlatButton(actions, "Aggiungi file", self.add_files,
                   kind="secondary").pack(side="left")
        FlatButton(actions, "Rimuovi", self.remove_selected,
                   kind="secondary").pack(side="left", padx=(8, 0))
        FlatButton(actions, "Svuota coda", self.clear_queue,
                   kind="secondary").pack(side="left", padx=(8, 0))
        FlatButton(actions, "Apri cartella output", self.open_output_dir,
                   kind="secondary").pack(side="right")

    def _build_action_bar(self, parent):
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x", pady=(0, 16))
        self.start_btn = FlatButton(bar, "Avvia coda", self.toggle_queue,
                                    kind="primary")
        self.start_btn.configure(font=(FONT, 11, "bold"), padx=24, pady=12)
        self.start_btn.pack(fill="x")

    def _build_status(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="x", pady=(0, 10))
        self.status_lbl = tk.Label(wrap, text="Pronto",
                                   font=(FONT, 10), fg=TXT, bg=BG, anchor="w")
        self.status_lbl.pack(fill="x")
        self.progress = ttk.Progressbar(
            wrap, mode="determinate", maximum=100,
            style="Verb.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(8, 4))
        self.pct_lbl = tk.Label(wrap, text="", font=(FONT, 9),
                                fg=TXT_MUTED, bg=BG, anchor="w")
        self.pct_lbl.pack(fill="x")

    def _build_preview(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text="ANTEPRIMA TRASCRIZIONE",
                 font=(FONT, 8, "bold"), fg=TXT_MUTED, bg=BG).pack(anchor="w")
        text_frame = tk.Frame(wrap, bg=SURFACE,
                              highlightthickness=1, highlightbackground=BORDER)
        text_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.transcript = tk.Text(
            text_frame, height=8, font=("Consolas", 10), wrap="word",
            state="disabled", bg=SURFACE, fg=TXT,
            insertbackground=TXT, relief="flat", borderwidth=0,
            padx=12, pady=10,
        )
        vsb = ttk.Scrollbar(text_frame, command=self.transcript.yview,
                            style="Verb.Vertical.TScrollbar")
        self.transcript.config(yscrollcommand=vsb.set)
        self.transcript.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    # ---------- helpers UI ----------

    def _set_status(self, text, color=TXT):
        self.status_lbl.config(text=text, fg=color)

    def _set_progress(self, msg, pct):
        self.pct_lbl.config(text=msg)
        if pct >= 0:
            try:
                self.progress.stop()
            except Exception:
                pass
            self.progress["mode"] = "determinate"
            self.progress["value"] = pct
        else:
            self.progress["mode"] = "indeterminate"
            try:
                self.progress.start(10)
            except Exception:
                pass

    def _set_transcript_lines(self, lines):
        self.transcript.config(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.insert("end", "".join(l + "\n" for l in lines))
        self.transcript.see("end")
        self.transcript.config(state="disabled")

    def _append_transcript(self, line):
        self.transcript.config(state="normal")
        self.transcript.insert("end", line + "\n")
        self.transcript.see("end")
        self.transcript.config(state="disabled")

    def _refresh_model_status(self):
        if model_exists():
            self.model_lbl.config(text="Modello Whisper Large-v3 pronto", fg=OK)
            try:
                self.download_btn.pack_forget()
            except Exception:
                pass
        else:
            self.model_lbl.config(
                text=f"Modello assente — scaricalo (~{MODEL_SIZE_GB:.1f} GB)",
                fg=WARN,
            )
            try:
                self.download_btn.pack(anchor="e", pady=(6, 0))
            except Exception:
                pass

    def show_components_ui(self):
        """Pannello diagnostica: cosa è installato e cosa userà l'app."""
        win = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} — Stato componenti")
        win.configure(bg=BG)
        win.geometry("560x500")
        win.minsize(480, 420)
        win.transient(self.root)

        pad = tk.Frame(win, bg=BG, padx=20, pady=18)
        pad.pack(fill="both", expand=True)

        tk.Label(pad, text="Stato componenti", font=(FONT, 16, "bold"),
                 fg=TXT, bg=BG).pack(anchor="w")

        cfg = decide_runtime_config(self.device_override.get())
        dev_txt = (f"GPU · {cfg['gpu_name']}" if cfg["device"] == "cuda" else "CPU")
        tk.Label(pad, text=f"Userà: {dev_txt}  ·  {cfg['mode']}",
                 font=(FONT, 11, "bold"), fg=TXT, bg=BG).pack(anchor="w", pady=(2, 0))
        tk.Label(pad, text=cfg["reason"], font=(FONT, 9), fg=TXT_MUTED, bg=BG,
                 wraplength=500, justify="left").pack(anchor="w", pady=(0, 2))
        if cfg.get("ram_gb"):
            tk.Label(pad, text=f"Sistema: {cfg['cores']} core CPU · {cfg['ram_gb']} GB RAM",
                     font=(FONT, 9), fg=TXT_MUTED, bg=BG).pack(anchor="w")

        ov_row = tk.Frame(pad, bg=BG)
        ov_row.pack(anchor="w", pady=(8, 14))
        tk.Label(ov_row, text="Modalità:", font=(FONT, 9), fg=TXT_MUTED,
                 bg=BG).pack(side="left", padx=(0, 8))
        _ov_labels = {"auto": "Automatico", "gpu": "Forza GPU", "cpu": "Forza CPU"}
        _ov_inv = {v: k for k, v in _ov_labels.items()}
        ov_disp = tk.StringVar(value=_ov_labels[self.device_override.get()])
        ov_cb = ttk.Combobox(ov_row, textvariable=ov_disp, state="readonly",
                             style="Verb.TCombobox", width=14,
                             values=list(_ov_labels.values()))
        ov_cb.pack(side="left")

        def _on_ov(_e=None):
            self.device_override.set(_ov_inv[ov_disp.get()])
            win.destroy()
            self.show_components_ui()   # ricarica per riflettere la nuova scelta

        ov_cb.bind("<<ComboboxSelected>>", _on_ov)

        for c in gather_component_status():
            if c["ok"]:
                mark, col = "✓", OK
            elif c.get("neutral"):
                mark, col = "•", TXT_MUTED
            else:
                mark, col = "✗", WARN

            card = tk.Frame(pad, bg=SURFACE,
                            highlightthickness=1, highlightbackground=BORDER)
            card.pack(fill="x", pady=4)
            inner = tk.Frame(card, bg=SURFACE, padx=14, pady=10)
            inner.pack(fill="x")

            head = tk.Frame(inner, bg=SURFACE)
            head.pack(fill="x")
            tk.Label(head, text=mark, font=(FONT, 12, "bold"),
                     fg=col, bg=SURFACE, width=2).pack(side="left")
            tk.Label(head, text=c["label"], font=(FONT, 11, "bold"),
                     fg=TXT, bg=SURFACE).pack(side="left")

            if c.get("detail"):
                tk.Label(inner, text=c["detail"], font=(FONT, 9),
                         fg=(TXT_MUTED if c["ok"] else col), bg=SURFACE,
                         wraplength=480, justify="left").pack(
                             anchor="w", padx=(26, 0))
            tk.Label(inner, text=c["what"], font=(FONT, 9),
                     fg=TXT_MUTED, bg=SURFACE, wraplength=480,
                     justify="left").pack(anchor="w", padx=(26, 0), pady=(2, 0))

            if c["key"] == "model" and not c["ok"]:
                FlatButton(inner, "Scarica modello",
                           lambda w=win: (w.destroy(), self.download_model_ui()),
                           kind="secondary").pack(anchor="w", padx=(26, 0),
                                                  pady=(8, 0))

        btns = tk.Frame(pad, bg=BG)
        btns.pack(fill="x", pady=(14, 0))
        FlatButton(btns, "Apri log di esecuzione", self._open_log_folder,
                   kind="secondary").pack(side="left")
        FlatButton(btns, "Chiudi", win.destroy,
                   kind="secondary").pack(side="right")

    def _open_log_folder(self):
        """Apre Esplora risorse evidenziando il file di log, da inviare per debug."""
        path = get_log_path()
        try:
            if os.path.isfile(path):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            else:
                os.startfile(get_base_path())
        except Exception as e:
            messagebox.showinfo(APP_NAME, f"Log in:\n{path}\n\n({e})")

    def download_model_ui(self):
        if model_exists():
            messagebox.showinfo(APP_NAME, "Modello già presente.")
            self._refresh_model_status()
            return
        self.download_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")

        def progress(msg, pct):
            self.root.after(0, self._set_progress, msg, pct)

        def worker():
            self.root.after(0, self._set_status, "Download modello in corso...", ACCENT)
            ok, err = download_model(progress)

            def finish():
                try:
                    self.progress.stop()
                except Exception:
                    pass
                self.progress["mode"] = "determinate"
                self.download_btn.configure(state="normal")
                self.start_btn.configure(state="normal")
                self._refresh_model_status()
                if ok:
                    self.progress["value"] = 100
                    self._set_status("Modello scaricato", OK)
                    messagebox.showinfo(
                        APP_NAME,
                        "Modello Whisper Large-v3 scaricato in models/.\n\n"
                        "Per uso offline puoi copiare la cartella su chiavetta.")
                else:
                    self.progress["value"] = 0
                    self._set_status("Download fallito", ERR)
                    messagebox.showerror(APP_NAME, f"Download fallito:\n{err}")
            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_queue_view(self):
        self.tree.delete(*self.tree.get_children())
        self.row_to_index.clear()
        for idx, it in enumerate(self.items):
            badge, _ = STATE_BADGE[it.state]
            iid = self.tree.insert(
                "", "end",
                values=(badge, os.path.basename(it.path), fmt_duration(it.duration)),
                tags=(it.state,),
            )
            self.row_to_index[iid] = idx
        pending = sum(1 for i in self.items if i.state == ST_PENDING)
        done = sum(1 for i in self.items if i.state == ST_DONE)
        total = len(self.items)
        self.queue_count.config(
            text=f"{total} file  ·  {pending} in attesa  ·  {done} completati"
        )

    def _editor_style(self):
        return {"bg": BG, "surface": SURFACE, "surface2": SURFACE_2, "txt": TXT,
                "txt_muted": TXT_MUTED, "accent": ACCENT, "border": BORDER,
                "ok": OK, "warn": WARN, "font": FONT}

    def _on_queue_double_click(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        idx = self.row_to_index.get(iid)
        if idx is None:
            return
        it = self.items[idx]
        if it.state != ST_DONE:
            messagebox.showinfo(
                APP_NAME, "Trascrivi prima questo file, poi potrai correggerlo.")
            return
        txt = os.path.splitext(it.path)[0] + ".txt"
        if not os.path.isfile(txt):
            messagebox.showwarning(APP_NAME, "File .txt della trascrizione non trovato.")
            return
        try:
            import transcript_editor
            log_event(f"Editor aperto: {os.path.basename(it.path)}")
            transcript_editor.open_editor(
                self.root, it.path, txt, style=self._editor_style(),
                on_saved=lambda p: log_event(
                    f"Trascrizione modificata e salvata: {os.path.basename(p)}"))
        except Exception as e:
            log_exception("Apertura editor trascrizione fallita", e)
            messagebox.showerror(APP_NAME, f"Impossibile aprire l'editor:\n{e}")

    def _update_item_state(self, idx, state):
        self.items[idx].state = state
        for iid, i in self.row_to_index.items():
            if i == idx:
                badge, _ = STATE_BADGE[state]
                self.tree.set(iid, "state", badge)
                self.tree.item(iid, tags=(state,))
                self.tree.see(iid)
                break

    # ---------- gestione coda ----------

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Aggiungi file audio alla coda",
            filetypes=[
                ("Audio", " ".join("*" + e for e in AUDIO_EXTS)),
                ("Tutti i file", "*.*"),
            ],
        )
        if not paths:
            return
        existing = {it.path for it in self.items}
        added = 0
        for p in paths:
            if p in existing:
                continue
            self.items.append(QueueItem(p))
            added += 1
        if added:
            self._refresh_queue_view()
            self._set_status(f"Aggiunti {added} file alla coda")

    def remove_selected(self):
        if self.queue_running:
            return
        sel = self.tree.selection()
        if not sel:
            return
        indices = sorted({self.row_to_index[iid] for iid in sel}, reverse=True)
        for i in indices:
            del self.items[i]
        self._refresh_queue_view()

    def clear_queue(self):
        if self.queue_running:
            return
        if not self.items:
            return
        if not messagebox.askyesno(APP_NAME, "Svuotare la coda?"):
            return
        self.items.clear()
        self._refresh_queue_view()
        self._set_status("Coda svuotata")

    def open_output_dir(self):
        # apre la cartella del primo file (o cwd)
        if self.items:
            d = os.path.dirname(self.items[0].path)
        else:
            d = get_base_path()
        try:
            os.startfile(d)
        except Exception:
            pass

    # ---------- avvio coda ----------

    def toggle_queue(self):
        if self.queue_running:
            # richiesta stop: termina dopo il file corrente
            self.stop_requested = True
            self.start_btn.configure(text="Arresto richiesto…", state="disabled")
            self._set_status("Mi fermo dopo il file corrente", WARN)
            return
        self.start_queue()

    def start_queue(self):
        pending = [it for it in self.items if it.state == ST_PENDING]
        if not pending:
            messagebox.showinfo(APP_NAME, "Nessun file in attesa nella coda.")
            return

        # Policy globale per file con .txt esistente
        with_existing = [it for it in pending
                         if os.path.isfile(os.path.splitext(it.path)[0] + ".txt")
                         and os.path.getsize(os.path.splitext(it.path)[0] + ".txt") > 0]
        policy = "fresh"  # default
        if with_existing:
            n = len(with_existing)
            ans = messagebox.askyesnocancel(
                APP_NAME,
                f"Per {n} file esiste già una trascrizione parziale.\n\n"
                "Sì  =  riprendi da dove avevi lasciato\n"
                "No  =  ricomincia da capo\n"
                "Annulla  =  salta quei file",
            )
            if ans is None:
                policy = "skip"
            elif ans:
                policy = "resume"
            else:
                policy = "fresh"

        for it in pending:
            txt = os.path.splitext(it.path)[0] + ".txt"
            has_existing = os.path.isfile(txt) and os.path.getsize(txt) > 0
            if has_existing and policy == "resume":
                it.resume_from = last_transcribed_time(txt)
            elif has_existing and policy == "skip":
                it.state = ST_SKIPPED
            else:
                it.resume_from = 0.0
        self._refresh_queue_view()

        self.queue_running = True
        self.stop_requested = False
        if hasattr(self, "_first_err_shown"):
            del self._first_err_shown
        self.start_btn.set_kind("danger")
        self.start_btn.configure(text="Ferma coda")

        threading.Thread(target=self._run_queue, daemon=True).start()

    def _run_queue(self):
        pending = [(i, it) for i, it in enumerate(self.items) if it.state == ST_PENDING]
        total = len(pending)

        # Auto-configurazione in base all'hardware (o all'override scelto).
        cfg = decide_runtime_config(self.device_override.get())
        self._runtime_config = cfg
        log_session_header(cfg)
        self.root.after(0, self._set_status, f"Configurazione: {cfg['reason']}", ACCENT)

        # Ricarica il modello se non è in cache o se il device scelto è cambiato
        # (es. l'utente ha messo "Forza CPU", oppure si rivuole la GPU dopo un
        # ripiego). Senza questo, un modello cachato sul device sbagliato verrebbe
        # riusato ignorando la nuova scelta.
        if self.whisper_model is not None and self._model_device != cfg["device"]:
            prev = self._model_device
            self.whisper_model = None
            if prev == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        if self.whisper_model is None:
            dev_lbl = "GPU" if cfg["device"] == "cuda" else "CPU"
            self.root.after(0, self._set_status,
                            f"Caricamento modello su {dev_lbl}...", ACCENT)
            self.root.after(0, self._set_progress, "Caricamento modello...", -1)
            try:
                if not model_exists():
                    self.root.after(0, self._set_status, "Download modello...", ACCENT)
                    ok, err = download_model(
                        lambda msg, pct: self.root.after(0, self._set_progress, msg, pct))
                    if not ok:
                        self.root.after(0, messagebox.showerror, APP_NAME,
                                        f"Download modello fallito:\n{err}")
                        log_event(f"Download modello fallito: {err}", level="ERROR")
                        self.root.after(0, self._finish_queue)
                        return
                import whisper
                self.whisper_model = whisper.load_model(
                    MODEL_NAME, download_root=get_models_dir(), device=cfg["device"])
                self._model_device = cfg["device"]
            except Exception as e:
                self.root.after(0, messagebox.showerror, APP_NAME,
                                f"Errore caricamento modello:\n{e}")
                log_exception("Caricamento modello (coda) fallito", e)
                self.root.after(0, self._finish_queue)
                return

        for n, (idx, it) in enumerate(pending, start=1):
            if self.stop_requested:
                break
            self.current_idx = idx
            self.root.after(0, self._update_item_state, idx, ST_RUNNING)
            self.root.after(0, self._set_status,
                            f"Elaborazione {n} di {total}  ·  {os.path.basename(it.path)}",
                            ACCENT)
            self.root.after(0, self._set_transcript_lines, [])

            done_event = threading.Event()
            result_holder = {"ok": False, "out": None, "info": None}

            def on_status(msg):
                self.root.after(0, self._set_status, msg, ACCENT)

            def on_progress(msg, pct):
                self.root.after(0, self._set_progress, msg, pct)

            def on_segment(line):
                if line == "__RELOAD__":
                    out = os.path.splitext(it.path)[0] + ".txt"
                    try:
                        with open(out, "r", encoding="utf-8") as f:
                            content = f.read().splitlines()
                        self.root.after(0, self._set_transcript_lines, content)
                    except Exception:
                        pass
                    return
                self.root.after(0, self._append_transcript, line)

            def on_done(ok, result, info):
                result_holder["ok"] = ok
                result_holder["out"] = result
                result_holder["info"] = info
                done_event.set()

            try:
                transcribe(
                    it.path,
                    (on_status, on_progress, on_segment, on_done),
                    resume_from=it.resume_from,
                    diarize_on=self.diar_var.get(),
                    num_speakers=(None if self.speakers_var.get() == "auto"
                                  else int(self.speakers_var.get())),
                    normalize_on=self.normalize_var.get(),
                    preloaded_model=self.whisper_model,
                    runtime_config=self._runtime_config,
                    on_fallback=self._on_device_fallback,
                )
                # transcribe ora è bloccante e chiama on_done internamente
                done_event.wait(timeout=1)
            except Exception as e:
                result_holder["ok"] = False
                result_holder["info"] = str(e)
                log_exception(f"Errore inatteso su {os.path.basename(it.path)}", e)

            if result_holder["ok"]:
                log_event(f"OK: {os.path.basename(it.path)} trascritto")
                self.root.after(0, self._update_item_state, idx, ST_DONE)
            else:
                err_msg = result_holder.get("out") or result_holder.get("info") or "errore sconosciuto"
                it.error = str(err_msg)
                log_event(f"ERRORE su {it.path}: {err_msg}", level="ERROR")
                self.root.after(0, self._update_item_state, idx, ST_ERROR)
                # mostra il primo errore SUBITO per capire cosa è successo
                if not hasattr(self, "_first_err_shown"):
                    self._first_err_shown = True
                    self.root.after(0, messagebox.showerror, APP_NAME,
                                    f"Errore su {os.path.basename(it.path)}:\n\n{err_msg}")

        self.root.after(0, self._finish_queue)

    def _on_device_fallback(self, new_model, new_cfg):
        """Chiamato da transcribe quando ripiega su CPU per OOM: aggiorna il
        modello in cache e la config, così i file successivi della coda usano
        direttamente la CPU senza ritentare la GPU."""
        self.whisper_model = new_model
        self._model_device = new_cfg.get("device", "cpu")
        self._runtime_config = new_cfg
        self.root.after(0, self._set_status,
                        "Passato a CPU: memoria GPU insufficiente per large-v3.", WARN)

    def _finish_queue(self):
        self.queue_running = False
        self.current_idx = None
        try:
            self.progress.stop()
        except Exception:
            pass
        self.progress["mode"] = "determinate"

        done = sum(1 for i in self.items if i.state == ST_DONE)
        errors = sum(1 for i in self.items if i.state == ST_ERROR)
        skipped = sum(1 for i in self.items if i.state == ST_SKIPPED)

        if self.stop_requested:
            self._set_status(f"Coda fermata  ·  {done} completati, {errors} errori",
                             WARN)
        elif errors == 0:
            self._set_status(f"Tutto completato  ·  {done} file trascritti",
                             OK)
        else:
            self._set_status(f"Completato con errori  ·  {done} ok, {errors} ko",
                             WARN)
        self.pct_lbl.config(text="")
        self.start_btn.set_kind("primary")
        self.start_btn.configure(text="Avvia coda", state="normal")
        self._refresh_queue_view()


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
            print(f"Trascrizione: {audio_path}")
            if not model_exists():
                print("Modello non trovato, download...")
                ok, err = download_model(lambda msg, _pct: print(f"\r{msg}", end="", flush=True))
                print()
                if not ok:
                    print(f"Errore download: {err}")
                    sys.exit(1)
            print("Caricamento modello...")
            import whisper
            cfg = decide_runtime_config("auto")
            print(f"Device: {cfg['device']} (fp16={cfg['fp16']}) — {cfg['reason']}")
            model = whisper.load_model(MODEL_NAME, download_root=get_models_dir(),
                                       device=cfg["device"])
            print("Trascrizione in corso...")
            result = model.transcribe(audio_path, language=None, fp16=cfg["fp16"])
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
