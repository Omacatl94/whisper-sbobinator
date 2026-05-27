# Sbobinator v2 — Diarization (riconoscimento parlanti) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere a Sbobinator il riconoscimento di chi parla (etichette "Interlocutore N" rinominabili), opzionale, completamente offline, su CPU.

**Architecture:** Si mantiene openai-whisper per la trascrizione (nessuna regressione). Quando la casella è accesa, dopo la trascrizione si esegue pyannote.audio sull'intero audio per ottenere i turni di parola, poi si abbina ogni segmento alla voce con massima sovrapposizione e si riscrive il `.txt` con le etichette. I modelli pyannote sono scaricati in build con un token HuggingFace e impacchettati nell'exe per il funzionamento offline.

**Tech Stack:** Python 3.12, tkinter, openai-whisper, **pyannote.audio + torchaudio** (nuovi), PyInstaller, ffmpeg.

**Branch:** `v2-diarization` (merge su `main` solo a v2 verificata).

**BLOCCO INIZIALE:** i Task ≥2 richiedono un **token HuggingFace** dell'utente con i termini accettati per i modelli `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, `pyannote/wespeaker-voxceleb-resnet34-LM`. Senza token non si scaricano i modelli.

---

### Task 1: Installare le dipendenze nel venv

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Installare pyannote.audio e torchaudio nel venv**

Run:
```
.\.venv\Scripts\python.exe -m pip install "pyannote.audio==3.3.2" torchaudio
```
Nota: `torchaudio` deve combaciare con il `torch` già installato; se pip propone un downgrade di torch, accettarlo è ok (resta CPU).

- [ ] **Step 2: Verificare l'import**

Run:
```
.\.venv\Scripts\python.exe -c "import pyannote.audio, torchaudio; from pyannote.audio import Pipeline; print('pyannote OK', pyannote.audio.__version__)"
```
Expected: stampa "pyannote OK 3.3.2" senza errori.

- [ ] **Step 3: Aggiornare requirements.txt**

Aggiungere le righe:
```
pyannote.audio==3.3.2
torchaudio
```

- [ ] **Step 4: Commit**

```
git add requirements.txt
git commit -m "v2: aggiunge dipendenze pyannote.audio + torchaudio"
```

---

### Task 2 (SPIKE): Diarizzare una clip a 2 voci con token HF (non-frozen)

Valida che pyannote funzioni e produca turni sensati. Richiede il token HF.

**Files:**
- Create: `spike_diarize.py` (temporaneo, non committato)

- [ ] **Step 1: Creare una clip a 2 voci**

Run (usa un audio con due parlanti, es. il calabrese):
```
.\dist\ffmpeg.exe -i "C:\Users\Omacatl\Desktop\Il pranzo calabrese con i parenti - Franco Neri.mp3" -t 60 -y clip2voci.wav
```

- [ ] **Step 2: Scrivere lo spike**

`spike_diarize.py`:
```python
import os, sys
TOKEN = os.environ["HF_TOKEN"]  # passato da riga di comando
from pyannote.audio import Pipeline
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=TOKEN)
import torch
pipe.to(torch.device("cpu"))
dia = pipe("clip2voci.wav")
turns = [(round(t.start, 1), round(t.end, 1), spk) for t, _, spk in dia.itertracks(yield_label=True)]
print("turni:", len(turns), "| parlanti:", sorted(set(s for *_, s in turns)))
for t in turns[:12]:
    print("  ", t)
```

- [ ] **Step 3: Eseguire (scarica i modelli, serve internet QUI)**

Run (PowerShell):
```
$env:HF_TOKEN="<TOKEN_UTENTE>"; .\.venv\Scripts\python.exe spike_diarize.py
```
Expected: stampa un numero di turni > 0 e ≥ 2 parlanti (`SPEAKER_00`, `SPEAKER_01`). Se errore di licenza → l'utente deve accettare i termini dei 3 modelli sul sito HF.

- [ ] **Step 4: Annotare dove sono stati scaricati i modelli**

Run:
```
.\.venv\Scripts\python.exe -c "import os; print(os.path.expanduser(os.path.join(os.environ.get('HF_HOME', '~/.cache/huggingface'), 'hub')))"
```
Annotare il percorso della cache `hub` (servirà per impacchettare offline al Task 9).

---

### Task 3: Verificare il caricamento OFFLINE dei modelli

Conferma che, scaricati i modelli, pyannote li carichi senza rete.

**Files:**
- Create: `spike_offline.py` (temporaneo)

- [ ] **Step 1: Scrivere lo spike offline**

`spike_offline.py`:
```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"          # vieta ogni accesso di rete
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pyannote.audio import Pipeline
import torch
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")  # senza token: usa la cache
pipe.to(torch.device("cpu"))
dia = pipe("clip2voci.wav")
spk = sorted(set(s for _, _, s in dia.itertracks(yield_label=True)))
print("OFFLINE OK | parlanti:", spk)
```

- [ ] **Step 2: Eseguire con HF_HUB_OFFLINE attivo**

Run:
```
.\.venv\Scripts\python.exe spike_offline.py
```
Expected: "OFFLINE OK | parlanti: [...]" senza errori di rete. Se fallisce, il caricamento offline va sistemato qui (prima di toccare l'app).

- [ ] **Step 3: Pulizia spike**

```
Remove-Item spike_diarize.py, spike_offline.py -ErrorAction SilentlyContinue
```

---

### Task 4: Funzione pura di abbinamento segmenti→voce (TDD)

La logica deterministica, testabile senza modelli.

**Files:**
- Create: `tests/test_assign.py`
- Modify: `sbobinator.py` (aggiunge `assign_speakers`)

- [ ] **Step 1: Scrivere il test che fallisce**

`tests/test_assign.py`:
```python
import sbobinator

def test_assign_picks_max_overlap():
    segments = [
        {"start": 0.0, "end": 4.0, "text": "ciao"},
        {"start": 4.0, "end": 8.0, "text": "salve"},
    ]
    turns = [(0.0, 3.5, "SPEAKER_00"), (3.5, 9.0, "SPEAKER_01")]
    out = sbobinator.assign_speakers(segments, turns)
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[1]["speaker"] == "SPEAKER_01"

def test_assign_no_turns_gives_none():
    segments = [{"start": 0.0, "end": 4.0, "text": "ciao"}]
    out = sbobinator.assign_speakers(segments, [])
    assert out[0]["speaker"] is None
```

- [ ] **Step 2: Eseguire il test (deve fallire)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_assign.py -v`
Expected: FAIL ("module 'sbobinator' has no attribute 'assign_speakers'").

- [ ] **Step 3: Implementare `assign_speakers` in sbobinator.py**

Aggiungere (vicino agli altri helper, prima di `transcribe`):
```python
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
```

- [ ] **Step 4: Eseguire il test (deve passare)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_assign.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```
git add sbobinator.py tests/test_assign.py
git commit -m "v2: assign_speakers (abbinamento segmento->voce) + test"
```

---

### Task 5: Etichette stabili e formato riga (TDD)

Mappa `SPEAKER_00→Interlocutore 1` in ordine di prima comparsa, e formatta la riga.

**Files:**
- Create: `tests/test_labels.py`
- Modify: `sbobinator.py`

- [ ] **Step 1: Test che fallisce**

`tests/test_labels.py`:
```python
import sbobinator

def test_label_mapping_by_first_appearance():
    segs = [
        {"start": 0, "end": 2, "text": "a", "speaker": "SPEAKER_01"},
        {"start": 2, "end": 4, "text": "b", "speaker": "SPEAKER_00"},
        {"start": 4, "end": 6, "text": "c", "speaker": "SPEAKER_01"},
    ]
    mapping = sbobinator.speaker_label_map(segs)
    assert mapping["SPEAKER_01"] == "Interlocutore 1"
    assert mapping["SPEAKER_00"] == "Interlocutore 2"

def test_format_line_with_speaker():
    seg = {"start": 65, "end": 70, "text": " ciao ", "speaker": "SPEAKER_00"}
    line = sbobinator.format_line(seg, {"SPEAKER_00": "Interlocutore 1"})
    assert line == "[01:05 - 01:10] Interlocutore 1: ciao"

def test_format_line_without_speaker():
    seg = {"start": 0, "end": 5, "text": "ciao", "speaker": None}
    line = sbobinator.format_line(seg, {})
    assert line == "[00:00 - 00:05] ciao"
```

- [ ] **Step 2: Eseguire (fallisce)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_labels.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementare in sbobinator.py**

```python
def speaker_label_map(segments):
    """SPEAKER_xx -> 'Interlocutore N' in ordine di prima comparsa."""
    mapping = {}
    for seg in segments:
        spk = seg.get("speaker")
        if spk and spk not in mapping:
            mapping[spk] = f"Interlocutore {len(mapping) + 1}"
    return mapping


def format_line(seg, label_map):
    s, e = seg["start"], seg["end"]
    ts = f"[{int(s // 60):02d}:{int(s % 60):02d} - {int(e // 60):02d}:{int(e % 60):02d}]"
    text = seg["text"].strip()
    spk = seg.get("speaker")
    if spk and spk in label_map:
        return f"{ts} {label_map[spk]}: {text}"
    return f"{ts} {text}"
```

- [ ] **Step 4: Eseguire (passa)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_labels.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```
git add sbobinator.py tests/test_labels.py
git commit -m "v2: mappa etichette Interlocutore N + format_line"
```

---

### Task 6: Modulo diarization (caricamento offline + run)

**Files:**
- Modify: `sbobinator.py`

- [ ] **Step 1: Aggiungere la funzione `diarize` in sbobinator.py**

```python
def diarize(audio_path, num_speakers=None, on_status=None):
    """Ritorna [(start, end, label)] usando pyannote, tutto offline e su CPU."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if on_status:
        on_status("Riconoscimento voci...")
    import torch
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipe.to(torch.device("cpu"))
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    dia = pipe(audio_path, **kwargs)
    return [(t.start, t.end, label) for t, _, label in dia.itertracks(yield_label=True)]
```

- [ ] **Step 2: Verifica manuale rapida (con clip)**

Run:
```
.\.venv\Scripts\python.exe -c "import sbobinator; print(len(sbobinator.diarize('clip2voci.wav')))"
```
Expected: stampa un intero > 0 (numero di turni). (Modelli già in cache dal Task 2.)

- [ ] **Step 3: Commit**

```
git add sbobinator.py
git commit -m "v2: funzione diarize (pyannote offline su CPU)"
```

---

### Task 7: Integrare la diarization nel flusso `transcribe()`

**Files:**
- Modify: `sbobinator.py` (funzione `transcribe`, firma e fase finale)

- [ ] **Step 1: Estendere la firma di `transcribe`**

Cambiare:
```python
def transcribe(audio_path, model_name, ui_callbacks, resume_from=0.0):
```
in:
```python
def transcribe(audio_path, model_name, ui_callbacks, resume_from=0.0,
               diarize_on=False, num_speakers=None):
```

- [ ] **Step 2: Dopo la trascrizione, prima del blocco finale, aggiungere la fase voci**

Subito dopo il blocco `finally:` che ripristina stdout/tqdm e chiude `out_file`, e prima del calcolo di `elapsed`, inserire:
```python
    if diarize_on:
        try:
            turns = diarize(audio_path, num_speakers=num_speakers, on_status=on_status)
            segs = assign_speakers(result["segments"], turns)
            label_map = speaker_label_map(segs)
            with open(out_path, "w", encoding="utf-8") as f:
                for seg in segs:
                    f.write(format_line(seg, label_map) + "\n")
            on_segment("__RELOAD__")  # segnala alla UI di ricaricare il file con le etichette
        except Exception as e:
            on_status(f"Voci non riconosciute: {e}")
```

- [ ] **Step 3: Gestire `__RELOAD__` nella UI (add_segment)**

In `App.add_segment`, all'inizio di `update()`:
```python
            if line == "__RELOAD__":
                self.transcript.config(state="normal")
                self.transcript.delete("1.0", "end")
                try:
                    with open(self._current_out, "r", encoding="utf-8") as f:
                        self.transcript.insert("end", f.read())
                except Exception:
                    pass
                self.transcript.config(state="disabled")
                return
```
(`self._current_out` va impostato in `start_transcription`, vedi Task 8.)

- [ ] **Step 4: Commit**

```
git add sbobinator.py
git commit -m "v2: fase diarization nel flusso transcribe + reload UI con etichette"
```

---

### Task 8: UI — casella, campo voci, passaggio parametri

**Files:**
- Modify: `sbobinator.py` (`App.__init__`, `pick_file`, `start_transcription`)

- [ ] **Step 1: Aggiungere i controlli in `__init__` (dopo il blocco modelli/radiobutton)**

```python
        diar_row = tk.Frame(frame, bg=CARA_BLU)
        diar_row.pack(fill="x", pady=(0, 10))
        self.diar_var = tk.BooleanVar(value=False)
        tk.Checkbutton(diar_row, text="Riconosci chi parla (più lento)",
                       variable=self.diar_var, bg=CARA_BLU, fg=CARA_TXT,
                       selectcolor=CARA_BLU_SCURO, activebackground=CARA_BLU,
                       activeforeground=CARA_ORO, highlightthickness=0).pack(side="left")
        tk.Label(diar_row, text="Voci attese:", font=("Arial", 9),
                 fg=CARA_TXT, bg=CARA_BLU).pack(side="left", padx=(12, 4))
        self.speakers_var = tk.StringVar(value="auto")
        ttk.Combobox(diar_row, textvariable=self.speakers_var, width=5, state="readonly",
                     values=["auto", "2", "3", "4", "5", "6"]).pack(side="left")
```

- [ ] **Step 2: In `start_transcription`, memorizzare out path e leggere i parametri**

Dopo `model_name = self.model_var.get()` aggiungere:
```python
        self._current_out = os.path.splitext(path)[0] + ".txt"
        diarize_on = self.diar_var.get()
        sp = self.speakers_var.get()
        num_speakers = None if sp == "auto" else int(sp)
```
E cambiare la riga del thread in:
```python
        t = threading.Thread(target=transcribe,
                             args=(path, model_name, callbacks, resume_from, diarize_on, num_speakers),
                             daemon=True)
```

- [ ] **Step 3: Verifica costruzione GUI (headless)**

Run:
```
.\.venv\Scripts\python.exe -c "import tkinter,sbobinator; r=tkinter.Tk(); r.withdraw(); sbobinator.App(r); print('GUI OK'); r.destroy()"
```
Expected: "GUI OK".

- [ ] **Step 4: Commit**

```
git add sbobinator.py
git commit -m "v2: UI casella diarization + campo voci + passaggio parametri"
```

---

### Task 9: Pulsante "Rinomina voci"

**Files:**
- Modify: `sbobinator.py` (`App`)

- [ ] **Step 1: Aggiungere il pulsante in `__init__` (nella btn_row o sotto)**

```python
        self.rename_btn = tk.Button(diar_row, text="Rinomina voci", command=self.rename_speakers,
                                    font=("Arial", 9), bg=CARA_BLU_SCURO, fg=CARA_TXT,
                                    activebackground=CARA_ORO, activeforeground=CARA_BLU,
                                    relief="flat", state="disabled")
        self.rename_btn.pack(side="right")
```

- [ ] **Step 2: Abilitare il pulsante in `on_done` quando la diarization è andata**

In `on_done`, dentro il ramo `if success:` aggiungere:
```python
                    if "Interlocutore" in self.transcript.get("1.0", "end"):
                        self.rename_btn.config(state="normal")
```

- [ ] **Step 3: Implementare `rename_speakers`**

```python
    def rename_speakers(self):
        text = self.transcript.get("1.0", "end")
        import re
        found = sorted(set(re.findall(r"Interlocutore \d+", text)),
                       key=lambda s: int(s.split()[1]))
        if not found:
            return
        win = tk.Toplevel(self.root)
        win.title("Rinomina voci")
        win.configure(bg=CARA_BLU)
        entries = {}
        for i, name in enumerate(found):
            tk.Label(win, text=name + " →", bg=CARA_BLU, fg=CARA_TXT).grid(row=i, column=0, padx=8, pady=4, sticky="e")
            e = tk.Entry(win, width=24)
            e.grid(row=i, column=1, padx=8, pady=4)
            entries[name] = e

        def apply():
            mapping = {old: e.get().strip() for old, e in entries.items() if e.get().strip()}
            if mapping:
                # sostituzione nel file
                try:
                    with open(self._current_out, "r", encoding="utf-8") as f:
                        content = f.read()
                    for old, new in mapping.items():
                        content = content.replace(old + ":", new + ":")
                    with open(self._current_out, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    pass
                # sostituzione nella finestra
                self.transcript.config(state="normal")
                box = self.transcript.get("1.0", "end")
                for old, new in mapping.items():
                    box = box.replace(old + ":", new + ":")
                self.transcript.delete("1.0", "end")
                self.transcript.insert("end", box)
                self.transcript.config(state="disabled")
            win.destroy()

        tk.Button(win, text="Applica", command=apply).grid(row=len(found), column=0, columnspan=2, pady=8)
```

- [ ] **Step 4: Verifica costruzione GUI (headless)**

Run:
```
.\.venv\Scripts\python.exe -c "import tkinter,sbobinator; r=tkinter.Tk(); r.withdraw(); a=sbobinator.App(r); print('OK', hasattr(a,'rename_speakers')); r.destroy()"
```
Expected: "OK True".

- [ ] **Step 5: Commit**

```
git add sbobinator.py
git commit -m "v2: pulsante Rinomina voci (finestra + sostituzione file/finestra)"
```

---

### Task 10: Pacchettizzazione offline + build + verifica nell'exe

**Files:**
- Modify: `sbobinator.spec`

- [ ] **Step 1: Copiare la cache modelli pyannote nel repo per impacchettarla**

Run (usa il percorso annotato al Task 2):
```
New-Item -ItemType Directory -Force -Path hf_models | Out-Null
Copy-Item -Recurse "<PERCORSO_CACHE_HUB>\*" hf_models\
```

- [ ] **Step 2: Far puntare HF alla cache impacchettata, all'avvio (sbobinator.py, vicino al guard stdout)**

```python
if getattr(sys, "frozen", False):
    _hf = os.path.join(get_base_path(), "hf_models")
    if os.path.isdir(_hf):
        os.environ["HF_HOME"] = _hf
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf, "hub") if os.path.isdir(os.path.join(_hf, "hub")) else _hf
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
```
Nota: `get_base_path` è già definita; questo blocco va dopo la sua definizione o spostando il riferimento dentro `main`. In alternativa impostare HF_HOME in cima a `diarize()`.

- [ ] **Step 3: Aggiornare lo .spec per dati e import nascosti**

In `sbobinator.spec`:
```python
datas = collect_data_files('whisper') + collect_data_files('pyannote') \
        + collect_data_files('torchaudio') + [('hf_models', 'hf_models')]
hiddenimports = collect_submodules('whisper') + collect_submodules('pyannote') \
        + collect_submodules('torchaudio') + collect_submodules('asteroid_filterbanks') \
        + ['tiktoken', 'tiktoken_ext', 'tiktoken_ext.openai_public']
```

- [ ] **Step 4: Build**

Run: `.\.venv\Scripts\pyinstaller.exe sbobinator.spec --noconfirm --clean`
Expected: build completata, `dist\Sbobinator.exe` presente.

- [ ] **Step 5: VERIFICA OFFLINE nell'exe — copiare clip e lanciare via CLI**

(L'exe è windowed; per il test usare la diarization da un piccolo script che importa il modulo congelato non è possibile. Verifica invece: avviare l'exe, attivare la casella, selezionare `clip2voci.wav`, confermare che compaiono `Interlocutore 1/2`. Disconnettere la rete per provare l'offline.)

Checklist manuale:
- [ ] L'exe si apre col tema.
- [ ] Casella accesa + clip a 2 voci → al termine il testo ha `Interlocutore 1:`/`Interlocutore 2:`.
- [ ] Con rete disconnessa funziona uguale (offline confermato).
- [ ] "Rinomina voci" sostituisce le etichette in finestra e nel `.txt`.
- [ ] Casella spenta → comportamento identico alla v1.0 (nessuna regressione).

- [ ] **Step 6: Commit**

```
git add sbobinator.spec sbobinator.py
git commit -m "v2: pacchettizzazione offline modelli pyannote nello .spec + build"
```

---

### Task 11: Merge su main e release

- [ ] **Step 1: Eseguire tutti i test**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: tutti PASS.

- [ ] **Step 2: Merge su main (fa partire la CI che aggiorna la release v1.0)**

```
git checkout main
git merge v2-diarization
git push origin main
```

- [ ] **Step 3: Monitorare la build CI e confermare l'aggiornamento di Sbobinator.zip sulla release.**

Nota: la CI dovrà installare pyannote.audio e impacchettare `hf_models/` — verificare che `hf_models/` sia committato (non in .gitignore) o che la CI lo ricostruisca con un secret `HF_TOKEN`.

---

## Note di esecuzione

- **Ordine critico:** Task 1→3 validano la fattibilità (deps + offline). Se il Task 2 o 3 fallisce, fermarsi e rivedere l'approccio prima di proseguire.
- I Task 4 e 5 (funzioni pure) sono indipendenti dai modelli e si possono fare anche senza token.
- `hf_models/` può essere grande; valutare se committarlo nel repo o passarlo alla CI via secret. Decisione al Task 10/11.
