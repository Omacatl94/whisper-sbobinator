import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import whisper
import warnings

warnings.filterwarnings("ignore")

MODELS = {
    "Medium (~1.5GB, più veloce)": "medium",
    "Large (~3GB, più preciso, dialetti)": "large",
}


def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def transcribe(audio_path, model_name, progress_callback, done_callback):
    try:
        progress_callback("Caricamento modello...")
        cache_dir = os.path.join(get_base_path(), "models")
        os.makedirs(cache_dir, exist_ok=True)
        model = whisper.load_model(model_name, download_root=cache_dir)

        progress_callback("Sbobinatura in corso...")
        result = model.transcribe(audio_path, language=None, fp16=False)

        out_path = os.path.splitext(audio_path)[0] + ".txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                start = seg["start"]
                end = seg["end"]
                m1, s1 = int(start // 60), int(start % 60)
                m2, s2 = int(end // 60), int(end % 60)
                f.write(f"[{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}] {seg['text'].strip()}\n")

        detected = result.get("language", "?")
        done_callback(True, out_path, detected)
    except Exception as e:
        done_callback(False, str(e), None)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sbobinator")
        self.root.geometry("500x350")
        self.root.resizable(False, False)

        frame = tk.Frame(root, padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Sbobinator", font=("Arial", 18, "bold")).pack(pady=(0, 5))
        tk.Label(frame, text="Trascina o seleziona un file audio", font=("Arial", 10)).pack(pady=(0, 15))

        tk.Label(frame, text="Modello:", anchor="w").pack(fill="x")
        self.model_var = tk.StringVar(value=list(MODELS.keys())[0])
        model_menu = ttk.Combobox(frame, textvariable=self.model_var, values=list(MODELS.keys()), state="readonly")
        model_menu.pack(fill="x", pady=(0, 15))

        self.btn = tk.Button(frame, text="Seleziona file audio", command=self.pick_file, height=2)
        self.btn.pack(fill="x", pady=(0, 15))

        self.status = tk.Label(frame, text="Pronto", font=("Arial", 10), fg="gray")
        self.status.pack(pady=(0, 10))

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x")

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Seleziona file audio",
            filetypes=[
                ("Audio", "*.mp3 *.wav *.m4a *.ogg *.flac *.wma *.aac *.mp4 *.webm"),
                ("Tutti i file", "*.*"),
            ],
        )
        if path:
            self.start_transcription(path)

    def start_transcription(self, path):
        self.btn.config(state="disabled")
        self.progress.start(10)
        model_key = self.model_var.get()
        model_name = MODELS[model_key]

        def on_progress(msg):
            self.root.after(0, lambda: self.status.config(text=msg, fg="blue"))

        def on_done(success, result, lang):
            def update():
                self.progress.stop()
                self.btn.config(state="normal")
                if success:
                    self.status.config(text=f"Fatto! Lingua: {lang}", fg="green")
                    messagebox.showinfo("Sbobinator", f"Trascrizione salvata in:\n{result}")
                else:
                    self.status.config(text="Errore", fg="red")
                    messagebox.showerror("Errore", result)
            self.root.after(0, update)

        t = threading.Thread(target=transcribe, args=(path, model_name, on_progress, on_done), daemon=True)
        t.start()


def main():
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if os.path.isfile(audio_path):
            print(f"Sbobinatura: {audio_path}")
            print("Caricamento modello medium...")
            cache_dir = os.path.join(get_base_path(), "models")
            os.makedirs(cache_dir, exist_ok=True)
            model = whisper.load_model("medium", download_root=cache_dir)
            print("Sbobinatura in corso...")
            result = model.transcribe(audio_path, language=None, fp16=False)
            out_path = os.path.splitext(audio_path)[0] + ".txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for seg in result["segments"]:
                    start = seg["start"]
                    end = seg["end"]
                    m1, s1 = int(start // 60), int(start % 60)
                    m2, s2 = int(end // 60), int(end % 60)
                    f.write(f"[{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}] {seg['text'].strip()}\n")
            detected = result.get("language", "?")
            print(f"Fatto! Lingua rilevata: {detected}")
            print(f"Salvato in: {out_path}")
            return

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
