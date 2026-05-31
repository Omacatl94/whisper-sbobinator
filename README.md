# verbaLIA

Trascrittore audio locale basato su Whisper large-v3. Tutto offline: nessuna rete
dopo il download iniziale del modello.

## Installazione

L'app è distribuita sulla Release come **archivio in due parti** (il bundle supera
il limite di 2 GB per file di GitHub):

1. Scarica **entrambi** i file: `verbaLIA.7z.001` e `verbaLIA.7z.002`.
2. Mettili nella stessa cartella, tasto destro su `verbaLIA.7z.001` → estrai
   (con 7‑Zip o l'estrazione di Windows). Si ricompone in un'unica cartella.
3. Avvia `verbaLIA.exe`.

Per un PC **offline / air-gapped**: scarica ed estrai su un PC con internet,
scarica anche il modello dall'app (vedi sotto), poi copia l'intera cartella su
chiavetta verso il PC di destinazione.

## Uso
1. Avvia `verbaLIA.exe`.
2. Al primo avvio scarica il modello con **Scarica modello** (~3 GB, in `models/`).
3. Aggiungi uno o più file audio e avvia la coda.
4. Per ogni file viene salvato un `.txt` accanto all'audio.

## GPU / CPU automatico
L'app riconosce l'hardware e si configura da sola:
- **GPU NVIDIA** con VRAM sufficiente → usa la GPU (fp16).
- VRAM scarsa o **nessuna GPU** → usa la CPU.
- Se la memoria GPU finisce durante la trascrizione, ripiega automaticamente su CPU.

Dal pulsante **Stato componenti** vedi cosa è installato, la configurazione scelta,
un override manuale (Automatico / Forza GPU / Forza CPU) e il **log di esecuzione**
(`verbalia_debug.log`), utile da inviare per il debug.

## Note
- Le librerie CUDA sono già incluse nel bundle: non serve scaricare driver.
  Funziona comunque su PC senza NVIDIA (in modalità CPU).
