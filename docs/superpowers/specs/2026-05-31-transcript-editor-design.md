# verbaLIA — Editor transcript per-segmento (riascolta + correggi)

Data: 2026-05-31
Stato: design (approvato nei punti chiave; in attesa review spec)

## Obiettivo

Mentre rivede una trascrizione, Mirko deve poter **cliccare su un segmento**,
**riascoltare** quello spezzone audio e **correggere il testo** se Whisper ha
capito male. Per **ogni file** della coda, in modo indipendente. Uso forense:
il `.txt` corretto è il documento finale.

## Decisioni (dal brainstorming)

1. **Editor dedicato per file**, aperto dalla coda (doppio click su un file
   completato). La preview live resta invariata.
2. **Riascolto a 16 kHz mono** riusando l'audio che Whisper carica, riprodotto
   con `sounddevice`. Intelligibile e semplice; nessun file temporaneo.
3. **Si modifica solo il testo**; i timestamp restano quelli di Whisper.
4. Codice in un **modulo separato** `transcript_editor.py`.

## Architettura

Nuovo modulo `transcript_editor.py`, importato da `sbobinator.py`. PyInstaller lo
include automaticamente (è un import locale). Tre unità isolate:

### 1. Parsing / serializzazione (funzioni pure)

Formati di riga prodotti oggi da `format_line`:
- semplice: `[mm:ss - mm:ss] testo`
- diarizzato: `[mm:ss - mm:ss] Interlocutore N: testo`

- `parse_segments(text) -> list[dict]`: per ogni riga che matcha
  `^\[(\d{1,2}):(\d{2}) - (\d{1,2}):(\d{2})\]\s?(.*)$` ritorna
  `{"start": sec, "end": sec, "text": resto}`. Per le righe diarizzate il
  prefisso "Interlocutore N: " resta dentro `text` (modificabile come il resto).
  Le righe che non matchano (vuote) vengono ignorate.
- `serialize_segments(segments) -> str`: ricostruisce `[mm:ss - mm:ss] testo\n`
  con gli stessi timestamp. **Round-trip allineato**: `serialize(parse(x)) == x`
  per i file prodotti dall'app (stesso formato, singolo spazio, mm:ss zero-pad).

### 2. Player spezzone

- `load_audio_cached(path)`: carica una volta `whisper.audio.load_audio(path)`
  (float32 mono @ 16 kHz) e lo tiene in cache nell'editor.
- `play_segment(audio, start, end)`: `sounddevice.stop()` poi
  `sounddevice.play(audio[int(start*16000):int(end*16000)], 16000)`.
- Errori (nessun dispositivo audio, file audio mancante/spostato) → messaggio,
  mai crash.

### 3. Finestra editor (`Toplevel`)

`open_editor(root, audio_path, txt_path)`:
- Titolo = nome file. Area scrollabile (Canvas + frame interno) con una riga per
  segmento: **▶** (play di quel segmento) · `00:04-00:09` · `tk.Entry`/`Text`
  con il testo modificabile.
- Barra inferiore: **Salva** (scrive il `.txt` via `serialize_segments`),
  **Stop** (ferma il play), **Chiudi**.
- L'audio viene caricato in modo lazy al primo ▶ (con un breve "carico audio...").

## Flusso

Coda → doppio click su file **ST_DONE** → l'editor legge `splitext(path)[0]+".txt"`
→ `parse_segments` → righe → ▶ riascolto / correggo → **Salva** riscrive il `.txt`
accanto all'audio. Indipendente per ogni file.

## Wiring nell'App (`sbobinator.py`)

- Bind `<Double-1>` sul Treeview della coda → handler che apre l'editor solo se
  l'item selezionato è `ST_DONE` (altrimenti messaggio "trascrivi prima"); in
  caso di `.txt` mancante, messaggio.
- Nessun'altra modifica all'App.

## Gestione errori

- `.txt` mancante / audio spostato → messaggio, nessun crash.
- Nessun output audio → messaggio.
- File molto lungo (centinaia di segmenti) → Canvas scrollabile; accettabile.
  Migliaia di segmenti potrebbero rallentare la costruzione: noto il limite, non
  ottimizzo ora.
- Editor apribile solo su file completati → nessun conflitto con la coda in corso.

## Test

- Unit test su `parse_segments`/`serialize_segments`: round-trip su righe semplici
  e diarizzate; testo con `]`, `:` e caratteri accentati; righe vuote ignorate.
- Smoke test Tk dell'editor con root nascosto (costruzione senza errori).

## Fuori scope (YAGNI)

- Modifica di timestamp, unione/divisione di segmenti.
- Riascolto a qualità originale (ffmpeg).
- Precisione sub-secondo (i `.txt` salvano mm:ss; eventuale timing fine è un
  lavoro futuro separato).
