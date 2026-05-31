# Editor transcript per-segmento — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permettere di cliccare un segmento di una trascrizione completata per riascoltarne lo spezzone audio e correggerne il testo, per ogni file della coda.

**Architecture:** Nuovo modulo `transcript_editor.py` disaccoppiato dall'app (riceve colori/font via dict). Funzioni pure di parse/serialize, player a 16 kHz con `sounddevice`, finestra `Toplevel` con righe segmento. `sbobinator.py` aggiunge solo il binding del doppio click sulla coda.

**Tech Stack:** Python, tkinter/ttk, sounddevice, whisper.audio (caricamento 16 kHz), pytest.

---

### Task 1: Parse / serialize segmenti (funzioni pure)

**Files:**
- Create: `transcript_editor.py`
- Test: `tests/test_transcript_editor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transcript_editor.py
import transcript_editor as te


def test_parse_basic_line():
    segs = te.parse_segments("[00:00 - 00:04] Buongiorno a tutti\n")
    assert segs == [{"start": 0.0, "end": 4.0, "text": "Buongiorno a tutti"}]


def test_parse_diarized_keeps_speaker_in_text():
    segs = te.parse_segments("[01:05 - 01:09] Interlocutore 2: ci penso io\n")
    assert segs[0]["start"] == 65.0 and segs[0]["end"] == 69.0
    assert segs[0]["text"] == "Interlocutore 2: ci penso io"


def test_parse_ignores_non_matching_lines():
    assert te.parse_segments("riga libera\n\n[00:00 - 00:02] ok\n") == \
        [{"start": 0.0, "end": 2.0, "text": "ok"}]


def test_roundtrip_plain_and_diarized():
    text = ("[00:00 - 00:04] Buongiorno a tutti\n"
            "[00:04 - 00:09] Interlocutore 1: oggi parliamo\n")
    assert te.serialize_segments(te.parse_segments(text)) == text


def test_serialize_uses_edited_text_and_keeps_timestamps():
    segs = [{"start": 65.0, "end": 69.0, "text": "testo corretto"}]
    assert te.serialize_segments(segs) == "[01:05 - 01:09] testo corretto\n"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_transcript_editor.py -q`
Expected: FAIL (ModuleNotFoundError / AttributeError, `transcript_editor` non esiste).

- [ ] **Step 3: Implement parse/serialize in `transcript_editor.py`**

```python
"""Editor di una trascrizione: riascolto per segmento + correzione del testo.

Disaccoppiato da sbobinator: riceve i colori/font via `style`. Le funzioni di
parse/serialize sono pure e testabili a parte.
"""
import os
import re

import tkinter as tk
from tkinter import messagebox

# Righe prodotte da format_line: "[mm:ss - mm:ss] testo" oppure
# "[mm:ss - mm:ss] Interlocutore N: testo" (l'etichetta resta dentro il testo).
_LINE_RE = re.compile(r"^\[(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\]\s?(.*)$")


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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_transcript_editor.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add transcript_editor.py tests/test_transcript_editor.py
git commit -m "feat(editor): parse/serialize segmenti trascrizione (round-trip)"
```

---

### Task 2: Player dello spezzone audio (16 kHz)

**Files:**
- Modify: `transcript_editor.py` (aggiunge i player helper)
- Test: `tests/test_transcript_editor.py`

- [ ] **Step 1: Write the failing test (sounddevice mockato)**

```python
def test_play_segment_slices_and_plays(monkeypatch):
    import numpy as np
    import transcript_editor as te
    audio = np.arange(16000 * 10, dtype="float32")  # 10 s @ 16 kHz
    played = {}

    class FakeSD:
        @staticmethod
        def stop():
            played["stopped"] = True

        @staticmethod
        def play(data, sr):
            played["len"] = len(data)
            played["sr"] = sr

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", FakeSD)
    te.play_segment(audio, 2.0, 5.0)
    assert played["stopped"] is True
    assert played["sr"] == 16000
    assert played["len"] == 16000 * 3   # 3 secondi
```

- [ ] **Step 2: Run test, verify it fails**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_transcript_editor.py::test_play_segment_slices_and_plays -q`
Expected: FAIL (`play_segment` non definito).

- [ ] **Step 3: Implement player helpers in `transcript_editor.py`**

```python
SAMPLE_RATE = 16000


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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_transcript_editor.py::test_play_segment_slices_and_plays -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transcript_editor.py tests/test_transcript_editor.py
git commit -m "feat(editor): player dello spezzone audio a 16 kHz"
```

---

### Task 3: Finestra editor (`open_editor`)

**Files:**
- Modify: `transcript_editor.py` (aggiunge `open_editor`)
- Test: `tests/test_transcript_editor.py`

- [ ] **Step 1: Write the failing smoke test**

```python
def test_open_editor_builds_without_error(tmp_path):
    import tkinter as tk
    import transcript_editor as te
    txt = tmp_path / "a.txt"
    txt.write_text("[00:00 - 00:04] uno\n[00:04 - 00:08] due\n", encoding="utf-8")
    root = tk.Tk(); root.withdraw()
    try:
        win = te.open_editor(root, str(tmp_path / "a.wav"), str(txt))
        win.withdraw()
        root.update_idletasks(); root.update()
        assert win.winfo_exists()
    finally:
        root.destroy()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_transcript_editor.py::test_open_editor_builds_without_error -q`
Expected: FAIL (`open_editor` non definito).

- [ ] **Step 3: Implement `open_editor` in `transcript_editor.py`**

```python
_DEFAULT_STYLE = {
    "bg": "#0f1419", "surface": "#161b22", "surface2": "#1f242e",
    "txt": "#e6e8eb", "txt_muted": "#7d8590", "accent": "#4a9eff",
    "border": "#2a3142", "ok": "#3fb950", "warn": "#d29922", "font": "Segoe UI",
}


def open_editor(root, audio_path, txt_path, style=None):
    """Apre la finestra di revisione/correzione per un file completato.

    Ritorna la Toplevel (utile ai test). L'audio si carica al primo ▶.
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

    entries = []   # (seg, tk.Entry)
    for seg in segments:
        row = tk.Frame(rows, bg=st["bg"])
        row.pack(fill="x", pady=2)
        tk.Button(row, text="▶", width=3, relief="flat",
                  bg=st["surface2"], fg=st["txt"], activebackground=st["accent"],
                  command=lambda s=seg: play(s)).pack(side="left", padx=(0, 6))
        ts = f"{_fmt_ts(seg['start'])}-{_fmt_ts(seg['end'])}"
        tk.Label(row, text=ts, width=12, font=("Consolas", 9),
                 fg=st["txt_muted"], bg=st["bg"]).pack(side="left")
        var = tk.StringVar(value=seg["text"])
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_transcript_editor.py::test_open_editor_builds_without_error -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transcript_editor.py tests/test_transcript_editor.py
git commit -m "feat(editor): finestra revisione con play/edit/salva per segmento"
```

---

### Task 4: Doppio click dalla coda apre l'editor

**Files:**
- Modify: `sbobinator.py` (binding nel `_build_queue` + handler nell'App)

- [ ] **Step 1: Aggiungi il binding nel Treeview**

In `_build_queue`, subito dopo `self.tree.pack(side="left", fill="both", expand=True)`:

```python
        self.tree.bind("<Double-1>", self._on_queue_double_click)
```

- [ ] **Step 2: Aggiungi l'handler nell'App**

Aggiungi questo metodo nell'App (es. subito dopo `_refresh_queue_view`):

```python
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
            transcript_editor.open_editor(self.root, it.path, txt,
                                          style=self._editor_style())
        except Exception as e:
            log_exception("Apertura editor trascrizione fallita", e)
            messagebox.showerror(APP_NAME, f"Impossibile aprire l'editor:\n{e}")
```

- [ ] **Step 3: Verifica sintassi e import del modulo**

Run: `.venv/Scripts/python.exe -m py_compile sbobinator.py transcript_editor.py`
Expected: nessun errore.

- [ ] **Step 4: Smoke test del wiring (handler raggiungibile)**

Run:
```
PYTHONPATH=. .venv/Scripts/python.exe -c "import tkinter as tk, sbobinator as s; r=tk.Tk(); r.withdraw(); a=s.App(r); assert hasattr(a,'_on_queue_double_click'); print('wiring ok'); r.destroy()"
```
Expected: stampa `wiring ok`.

- [ ] **Step 5: Commit**

```bash
git add sbobinator.py
git commit -m "feat(editor): doppio click sulla coda apre l'editor del file completato"
```

---

### Task 5: Verifica finale

- [ ] **Step 1: Suite completa**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ -q`
Expected: tutti i test passano (compresi i nuovi di `test_transcript_editor.py`).

- [ ] **Step 2: Nota verifica manuale (dev-first, Paolo)**

Da exe/dev: trascrivi un file, doppio click sull'item completato → l'editor mostra i segmenti; ▶ riascolta lo spezzone; modifica un testo; Salva → il `.txt` riflette la modifica. Provare anche un file diarizzato.

---

## Note di verifica (self-review)

- Copertura spec: parse/serialize (Task 1), player 16 kHz (Task 2), finestra editor con play/edit/save (Task 3), doppio click per file completato (Task 4), test+manuale (Task 5). ✓
- Round-trip: `serialize_segments(parse_segments(text)) == text` testato su righe semplici e diarizzate.
- Nomi coerenti tra task: `parse_segments`, `serialize_segments`, `_fmt_ts`, `load_audio`, `play_segment`, `stop_playback`, `open_editor`, `_on_queue_double_click`.
- Disaccoppiamento: `transcript_editor` non importa `sbobinator`; riceve `style`; usa `whisper.audio`/`sounddevice` in import lazy → PyInstaller li include già (sono nel bundle).
