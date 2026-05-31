"""Editor di una trascrizione: riascolto per segmento + correzione del testo.

Disaccoppiato da sbobinator: riceve i colori/font via `style`. Le funzioni di
parse/serialize sono pure e testabili a parte. whisper.audio e sounddevice
vengono importati in modo lazy (sono comunque nel bundle PyInstaller).
"""
import os
import re

import tkinter as tk
from tkinter import messagebox

# Righe prodotte da format_line: "[mm:ss - mm:ss] testo" oppure
# "[mm:ss - mm:ss] Interlocutore N: testo" (l'etichetta resta dentro il testo).
_LINE_RE = re.compile(r"^\[(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\]\s?(.*)$")

SAMPLE_RATE = 16000


def parse_segments(text):
    """Testo del .txt -> lista di {start, end (secondi float), text}."""
    segs = []
    for line in text.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        sm, ss, em, es, body = m.groups()
        segs.append({
            "start": float(int(sm) * 60 + int(ss)),
            "end": float(int(em) * 60 + int(es)),
            "text": body,
        })
    return segs


def _fmt_ts(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def serialize_segments(segments):
    """Lista di segmenti -> testo del .txt, con gli stessi timestamp."""
    out = []
    for s in segments:
        out.append(f"[{_fmt_ts(s['start'])} - {_fmt_ts(s['end'])}] {s['text']}")
    return "".join(line + "\n" for line in out)


def load_audio(path):
    """Carica l'audio come float32 mono @ 16 kHz (riusa il loader di whisper)."""
    import whisper.audio
    return whisper.audio.load_audio(path)


def stop_playback():
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass


def play_segment(audio, start, end, sr=SAMPLE_RATE):
    """Riproduce la fetta [start, end] (secondi) dell'audio caricato."""
    import sounddevice as sd
    a = audio[int(start * sr):int(end * sr)]
    sd.stop()
    sd.play(a, sr)


_DEFAULT_STYLE = {
    "bg": "#0f1419", "surface": "#161b22", "surface2": "#1f242e",
    "txt": "#e6e8eb", "txt_muted": "#7d8590", "accent": "#4a9eff",
    "border": "#2a3142", "ok": "#3fb950", "warn": "#d29922", "font": "Segoe UI",
}


def open_editor(root, audio_path, txt_path, style=None, on_saved=None):
    """Apre la finestra di revisione/correzione per un file completato.

    Ritorna la Toplevel (utile ai test). L'audio si carica al primo play.
    on_saved(txt_path): callback opzionale invocato dopo un salvataggio riuscito
    (l'app lo usa per tracciare la modifica nel log di esecuzione).
    """
    st = dict(_DEFAULT_STYLE, **(style or {}))
    with open(txt_path, "r", encoding="utf-8") as f:
        segments = parse_segments(f.read())

    win = tk.Toplevel(root)
    win.title(f"Revisione — {os.path.basename(audio_path)}")
    win.configure(bg=st["bg"])
    win.geometry("720x600")
    win.minsize(560, 420)

    state = {"audio": None}   # cache lazy dell'audio

    def ensure_audio():
        if state["audio"] is None:
            try:
                state["audio"] = load_audio(audio_path)
            except Exception as e:
                messagebox.showwarning(win.title(),
                                       f"Audio non riproducibile:\n{e}")
                return None
        return state["audio"]

    def play(seg):
        a = ensure_audio()
        if a is None:
            return
        try:
            play_segment(a, seg["start"], seg["end"])
        except Exception as e:
            messagebox.showwarning(win.title(), f"Riproduzione fallita:\n{e}")

    pad = tk.Frame(win, bg=st["bg"], padx=16, pady=14)
    pad.pack(fill="both", expand=True)
    tk.Label(pad, text="Clicca ▶ per riascoltare, correggi il testo, poi Salva.",
             font=(st["font"], 10), fg=st["txt_muted"], bg=st["bg"]).pack(
                 anchor="w", pady=(0, 10))

    # Area scrollabile (Canvas + frame interno)
    canvas = tk.Canvas(pad, bg=st["bg"], highlightthickness=0)
    vsb = tk.Scrollbar(pad, orient="vertical", command=canvas.yview)
    rows = tk.Frame(canvas, bg=st["bg"])
    rows.bind("<Configure>",
              lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=rows, anchor="nw")
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    entries = []   # (seg, StringVar)
    for seg in segments:
        row = tk.Frame(rows, bg=st["bg"])
        row.pack(fill="x", pady=2)
        tk.Button(row, text="▶", width=3, relief="flat",
                  bg=st["surface2"], fg=st["txt"], activebackground=st["accent"],
                  command=lambda s=seg: play(s)).pack(side="left", padx=(0, 6))
        ts = f"{_fmt_ts(seg['start'])}-{_fmt_ts(seg['end'])}"
        tk.Label(row, text=ts, width=12, font=("Consolas", 9),
                 fg=st["txt_muted"], bg=st["bg"]).pack(side="left")
        var = tk.StringVar(master=win, value=seg["text"])
        ent = tk.Entry(row, textvariable=var, font=(st["font"], 10),
                       bg=st["surface"], fg=st["txt"], insertbackground=st["txt"],
                       relief="flat")
        ent.pack(side="left", fill="x", expand=True, padx=(6, 0))
        entries.append((seg, var))

    def save():
        edited = [{"start": s["start"], "end": s["end"], "text": v.get()}
                  for s, v in entries]
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(serialize_segments(edited))
            if on_saved:
                try:
                    on_saved(txt_path)
                except Exception:
                    pass
            messagebox.showinfo(win.title(), "Trascrizione salvata.")
        except Exception as e:
            messagebox.showwarning(win.title(), f"Salvataggio fallito:\n{e}")

    bar = tk.Frame(win, bg=st["bg"], padx=16, pady=10)
    bar.pack(fill="x")
    tk.Button(bar, text="Stop", relief="flat", bg=st["surface2"], fg=st["txt"],
              command=stop_playback).pack(side="left")
    tk.Button(bar, text="Chiudi", relief="flat", bg=st["surface2"], fg=st["txt"],
              command=win.destroy).pack(side="right")
    tk.Button(bar, text="Salva", relief="flat", bg=st["accent"], fg=st["bg"],
              command=save).pack(side="right", padx=(0, 8))

    win.protocol("WM_DELETE_WINDOW", lambda: (stop_playback(), win.destroy()))
    return win
