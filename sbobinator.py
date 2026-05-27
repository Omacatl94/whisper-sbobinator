import sys
import os
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import warnings

warnings.filterwarnings("ignore")

# In una build PyInstaller "windowed" (console=False) sys.stdout e sys.stderr
# valgono None. whisper/tqdm ci scrivono sopra durante la trascrizione e l'app
# crasha con "NoneType object has no attribute 'write'". Reindirizziamo i
# flussi mancanti su devnull così ogni scrittura va a vuoto senza errori.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

MODEL_INFO = {
    "medium": {"label": "Medium (~1.5GB, più veloce)", "file": "medium.pt", "size_gb": 1.5},
    "large": {"label": "Large (~3GB, più preciso, dialetti)", "file": "large-v3.pt", "size_gb": 2.9},
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
    return os.path.isfile(os.path.join(get_models_dir(), info["file"]))


def check_ffmpeg():
    ffmpeg_local = os.path.join(get_base_path(), "ffmpeg.exe")
    if os.path.isfile(ffmpeg_local):
        os.environ["PATH"] = get_base_path() + os.pathsep + os.environ.get("PATH", "")
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


def download_model(model_name, progress_callback):
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


def transcribe(audio_path, model_name, ui_callbacks, resume_from=0.0):
    on_status, on_progress, on_segment, on_done = ui_callbacks

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
        model = whisper.load_model(model_name, download_root=get_models_dir())
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
                 fg=CARA_ROSSO, bg=CARA_BLU).pack(pady=(0, 12))

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
                    messagebox.showinfo("Sbobinator", f"Trascrizione salvata in:\n{result}")
                else:
                    self.progress["value"] = 0
                    self.status.config(text="Errore", fg=COL_ERR)
                    self.pct_label.config(text="")
                    messagebox.showerror("Errore", result)
            self.root.after(0, update)

        callbacks = (on_status, on_progress, on_segment, on_done)
        t = threading.Thread(target=transcribe, args=(path, model_name, callbacks, resume_from), daemon=True)
        t.start()


def main():
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
            model = whisper.load_model(model_name, download_root=get_models_dir())
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
