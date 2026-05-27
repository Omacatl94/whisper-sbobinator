import sys
import os
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import warnings

warnings.filterwarnings("ignore")

MODEL_INFO = {
    "medium": {"label": "Medium (~1.5GB, più veloce)", "file": "medium.pt", "size_gb": 1.5},
    "large": {"label": "Large (~3GB, più preciso, dialetti)", "file": "large-v3.pt", "size_gb": 2.9},
}


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


def transcribe(audio_path, model_name, ui_callbacks):
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

    on_status("Sbobinatura in corso...")
    on_progress("Sbobinatura...", 0)

    start_time = time.time()

    try:
        result = whisper.transcribe(
            model, audio_path, language=None, fp16=False, verbose=False,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            on_done(False, "Memoria insufficiente!\n\nProva col modello Medium che è più leggero.", None)
        else:
            on_done(False, f"Errore durante la sbobinatura:\n{e}", None)
        return
    except Exception as e:
        on_done(False, f"Errore durante la sbobinatura:\n{e}", None)
        return

    on_status("Salvataggio trascrizione...")
    out_path = os.path.splitext(audio_path)[0] + ".txt"

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                start = seg["start"]
                end = seg["end"]
                m1, s1 = int(start // 60), int(end % 60)
                m2, s2 = int(end // 60), int(end % 60)
                text = seg["text"].strip()
                line = f"[{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}] {text}"
                f.write(line + "\n")
                on_segment(line)
    except PermissionError:
        on_done(False, f"Non riesco a scrivere il file:\n{out_path}\n\nControlla i permessi della cartella.", None)
        return
    except Exception as e:
        on_done(False, f"Errore salvataggio:\n{e}", None)
        return

    elapsed = time.time() - start_time
    detected = result.get("language", "?")
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    on_progress("Completato!", 100)
    on_done(True, out_path, f"{detected} — tempo: {mins}m {secs}s")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sbobinator")
        self.root.geometry("600x550")
        self.root.resizable(False, False)

        frame = tk.Frame(root, padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Sbobinator", font=("Arial", 18, "bold")).pack(pady=(0, 3))
        tk.Label(frame, text="Trascrittore audio locale", font=("Arial", 9), fg="gray").pack(pady=(0, 12))

        self.model_status = tk.Label(frame, text="", font=("Arial", 9))
        self.model_status.pack(pady=(0, 5))
        self.check_models()

        row = tk.Frame(frame)
        row.pack(fill="x", pady=(0, 10))
        tk.Label(row, text="Modello:", font=("Arial", 10)).pack(side="left")
        self.model_var = tk.StringVar(value="medium")
        for name, info in MODEL_INFO.items():
            tk.Radiobutton(row, text=info["label"], variable=self.model_var, value=name,
                           font=("Arial", 9), command=self.check_models).pack(side="left", padx=(10, 0))

        self.btn = tk.Button(frame, text="Seleziona file audio", command=self.pick_file,
                             height=2, font=("Arial", 11))
        self.btn.pack(fill="x", pady=(0, 10))

        self.status = tk.Label(frame, text="Pronto", font=("Arial", 10), fg="gray")
        self.status.pack(pady=(0, 5))

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 5))

        self.pct_label = tk.Label(frame, text="", font=("Arial", 9), fg="gray")
        self.pct_label.pack(pady=(0, 10))

        tk.Label(frame, text="Trascrizione live:", font=("Arial", 9, "bold"), anchor="w").pack(fill="x")
        text_frame = tk.Frame(frame)
        text_frame.pack(fill="both", expand=True, pady=(3, 0))

        self.transcript = tk.Text(text_frame, height=10, font=("Consolas", 9), wrap="word",
                                  state="disabled", bg="#f5f5f5")
        scrollbar = ttk.Scrollbar(text_frame, command=self.transcript.yview)
        self.transcript.config(yscrollcommand=scrollbar.set)
        self.transcript.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def check_models(self):
        model = self.model_var.get()
        if model_exists(model):
            self.model_status.config(text=f"✓ Modello {model} trovato", fg="green")
        else:
            self.model_status.config(
                text=f"✗ Modello {model} non trovato — verrà scaricato al primo uso (serve internet)",
                fg="orange")

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Seleziona file audio",
            filetypes=[
                ("Audio", "*.mp3 *.wav *.m4a *.ogg *.flac *.wma *.aac *.mp4 *.webm *.opus"),
                ("Tutti i file", "*.*"),
            ],
        )
        if path:
            self.start_transcription(path)

    def add_segment(self, line):
        def update():
            self.transcript.config(state="normal")
            self.transcript.insert("end", line + "\n")
            self.transcript.see("end")
            self.transcript.config(state="disabled")
        self.root.after(0, update)

    def start_transcription(self, path):
        self.btn.config(state="disabled")
        self.transcript.config(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.config(state="disabled")
        self.progress["value"] = 0

        model_name = self.model_var.get()

        def on_status(msg):
            self.root.after(0, lambda: self.status.config(text=msg, fg="blue"))

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
                    self.status.config(text=f"Fatto! {info}", fg="green")
                    self.pct_label.config(text="100%")
                    messagebox.showinfo("Sbobinator", f"Trascrizione salvata in:\n{result}")
                else:
                    self.progress["value"] = 0
                    self.status.config(text="Errore", fg="red")
                    self.pct_label.config(text="")
                    messagebox.showerror("Errore", result)
            self.root.after(0, update)

        callbacks = (on_status, on_progress, on_segment, on_done)
        t = threading.Thread(target=transcribe, args=(path, model_name, callbacks), daemon=True)
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
    root.mainloop()


if __name__ == "__main__":
    main()
