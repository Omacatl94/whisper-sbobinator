# verbaLIA — Fix distribuzione v3 + pannello diagnostica

Data: 2026-05-31
Stato: design (in attesa di approvazione Paolo)

## Problema

Mirko non riesce a scaricare la v3 dalla Release `v1.0`: ci sono i changelog ma
manca il file. Causa accertata dai log CI del commit `92c3cda`:

- Il build **funziona** (EXE compilato, bundle creato, artifact caricato).
- Fallisce **solo** lo step "Publish to release v1.0":
  ```
  Final size is 2995915571 bytes        (verbaLIA.zip ≈ 2,99 GB)
  HTTP 422: Validation Failed
  size must be less than 2147483648      (limite GitHub: 2 GiB per asset)
  ```
- Risultato: la release resta ferma a `Sbobinator.zip` del 27/05 (v1, 360 MB).

Il bundle è esploso da 360 MB (v1) a 3 GB (v3) perché la v3 compila **torch con
CUDA cu128** (per la 5070 Ti Blackwell di Paolo e la T1000 di Mirko). Le DLL CUDA
pesano ~3,4 GB e finiscono dentro `verbaLIA.exe` (build onefile).

## Idea iniziale (di Paolo) e perché NON è fattibile

Idea: come per il modello, tenere l'app leggera e far **scaricare le DLL CUDA
on-demand** dalla GUI solo quando servono.

**Verificato sperimentalmente che non è possibile** con la build cu128:

- `_load_dll_libraries()` di torch fa `glob("*.dll")` su `torch/lib` e **carica
  ogni DLL presente** all'import, fallendo se manca una dipendenza.
- Analisi delle tabelle di import (pefile): `torch_python.dll`, `torch.dll` e
  `shm.dll` importano **staticamente** `torch_cuda.dll`, che a sua volta importa
  staticamente cublas/cublasLt/cufft/cusparse/cusolver/nvJitLink/cudart.
- Chiusura delle dipendenze richieste **all'import** = **~2,45 GB** di DLL CUDA.
  Non rimovibili né rimandabili: `import torch` le pretende tutte subito.

Conclusione: da una singola build cu128 **non** si può ricavare una base
leggera togliendo CUDA. L'idea "scarico solo le DLL dopo" è bloccata da come
PyTorch è linkato, non da una nostra scelta.

## Scoperta che semplifica tutto

Nessuna DLL del bundle importa staticamente il driver `nvcuda.dll` (verificato
con pefile). Quindi **la build cu128 gira già ovunque**:

- su PC **senza** NVIDIA (es. il PC personale AMD di Mirko) → parte e usa la CPU
  (`torch.cuda.is_available()` → False, fallback automatico);
- su PC **con** NVIDIA (es. T1000 dell'ufficio) → usa la GPU.

L'unico vero ostacolo è il limite 2 GiB di GitHub. **Un solo bundle universale,
spezzato in volumi <2 GB, risolve tutto.**

## Design

### 1. Distribuzione: split del bundle (fix del bug)

- CI crea il bundle come **archivio 7z multi-volume** da 1800 MiB:
  `verbaLIA.7z.001`, `verbaLIA.7z.002` (ognuno < 2 GiB).
- Lo step di publish carica **tutti** i volumi sulla release con `--clobber`.
- L'utente scarica i volumi e li estrae (il primo volume tira gli altri).
- Il **modello** resta scaricato a parte dal CDN OpenAI (già così): NON entra nel
  bundle → niente raddoppio di peso. Portabilità air-gapped invariata.

Decisione aperta per Paolo: estrazione sul PC ufficio air-gapped.
- (a) 7z semplici → richiedono 7-Zip o Windows 11 recente.
- (b) volume auto-estraente (`-sfx`) → nessun software, ma possibile allarme
  SmartScreen/AV in caserma.
Default proposto: (a) 7z semplici + istruzioni chiare. Da confermare.

### 2. Pannello "Stato componenti" (diagnostica guidata)

Una finestra raggiungibile dall'header che mostra, per ogni componente:
stato ✓/✗, a cosa serve, e azione se manca. Riusa funzioni già esistenti
(`check_ffmpeg`, `model_exists`, `get_device_info`).

Voci:
- **ffmpeg** — "Legge audio/video. Necessario." (✓/✗)
- **Modello Whisper large-v3** — "Il motore della trascrizione. Obbligatorio."
  (✓/✗ + pulsante Scarica, già esistente)
- **GPU rilevata** — nome scheda + VRAM, o "nessuna GPU NVIDIA → uso CPU"
- **Accelerazione GPU attiva** — `torch.cuda.is_available()` (✓/✗) con spiegazione
  "Su questo PC la trascrizione userà: CPU / GPU <nome>"

Principi: informativo e user-friendly; ogni voce spiega **a cosa serve**; nessun
gating che nasconda informazioni in base all'hardware locale.

Nota: NON include il download on-demand delle DLL CUDA (non fattibile, vedi
sopra). Per la GPU il bundle le contiene già.

### 3. Pulizia

- README/Release: aggiornare il riferimento da `verbaLIA.zip` ai volumi
  `verbaLIA.7z.001/.002` + istruzioni di estrazione.

## Cosa NON facciamo (YAGNI)

- Niente edizione CPU separata: Mirko ha bisogno della GPU (T1000), e la build
  cu128 gira comunque su CPU dove serve. Una seconda edizione aggiungerebbe
  complessità senza servire al caso d'uso reale.
- Niente download on-demand delle DLL CUDA (non fattibile).

## Verifica

- CI: i due volumi vengono caricati sulla release senza errore 422.
- Locale (dev-first): Paolo scarica i volumi, estrae, l'app parte; il pannello
  mostra correttamente ffmpeg/modello/GPU; trascrizione GPU sul PC con NVIDIA.
- Niente push finché Paolo non conferma in locale.
