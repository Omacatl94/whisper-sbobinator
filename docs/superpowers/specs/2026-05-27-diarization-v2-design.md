# Sbobinator v2 — Riconoscimento dei parlanti (speaker diarization)

Data: 2026-05-27
Branch: `v2-diarization`

## Obiettivo

Aggiungere a Sbobinator il riconoscimento di **chi parla**: oltre a trascrivere,
etichettare ogni riga con la voce ("Interlocutore 1", "Interlocutore 2", …),
con possibilità di rinominare le voci con nomi reali.

## Vincoli (non negoziabili)

- **Offline totale**: il PC di destinazione (stazione Carabinieri) **non ha
  internet**. Tutti i modelli vanno scaricati in fase di build (su questo PC) e
  impacchettati nell'exe. A runtime: zero rete.
- **GPU AMD/ATI**: niente CUDA → niente accelerazione affidabile. Si gira su
  **CPU**. (torch-directml resta un'ipotesi futura, non garantita.)
- **PC debole**: le prestazioni contano. Per questo la diarization è opzionale.
- **Privacy**: nessun servizio cloud. Tutto in locale.

## Approccio scelto (A)

Mantenere il motore di trascrizione attuale (**openai-whisper**) e **aggiungere
pyannote.audio** come passo separato di diarization. Nessuna regressione: con la
funzione spenta l'app resta identica alla v1.0 (live, salvataggio progressivo,
riprendi). Approcci scartati: faster-whisper (riscrittura del cuore, rinviata a
eventuale upgrade di velocità) e WhisperX (dipendenze pesanti, meno controllo,
packaging offline più difficile).

## Requisiti funzionali

1. Casella **"Riconosci chi parla (più lento)"** — opzionale, default spenta.
2. Campo **"Voci attese"**: `auto / 2 / 3 / 4 / 5…` (default `auto`). Se l'utente
   sa quante voci ci sono, pyannote è più preciso e veloce.
3. Etichette **anonime e rinominabili**: `Interlocutore N`.
4. Pulsante **"Rinomina voci"** nell'interfaccia (vedi sotto).

## Flusso di lavoro

### Casella SPENTA
Identico alla v1.0. Nessun cambiamento, nessun rallentamento.

### Casella ACCESA
1. **Trascrizione** (come ora): whisper produce i segmenti, mostrati live e
   salvati man mano sul `.txt` **senza** etichette.
2. **Riconoscimento voci**: a trascrizione finita, pyannote analizza **l'intero
   audio** e produce i turni di parola `(inizio, fine, voce)`. Stato:
   "Riconoscimento voci…", barra in modalità animata.
3. **Abbinamento**: a ogni segmento di whisper si assegna la voce con **massima
   sovrapposizione temporale** con i turni di pyannote.
4. **Riscrittura finale**: `.txt` e finestra aggiornati con le etichette.

Trascrizione prima, voci dopo (sequenziale): su un PC debole evita di far girare
due modelli insieme.

### Formato output
```
[00:00 - 00:05] Interlocutore 1: Buongiorno, allora mi dica.
[00:05 - 00:09] Interlocutore 2: Sì, ecco, è successo che...
```

## Interfaccia

- Sotto la scelta del modello: casella "Riconosci chi parla (più lento)" e campo
  "Voci attese" (auto/N).
- Durante l'uso con voci attive, lo stato mostra le fasi: `Sbobinatura…` →
  `Riconoscimento voci…` → `Completato`.
- **Pulsante "Rinomina voci"**: attivo dopo una trascrizione con voci
  riconosciute. Apre una finestrella con l'elenco delle voci trovate e un campo
  nome per ciascuna; "Applica" sostituisce le etichette **sia nella finestra sia
  nel file `.txt`**.

## Crash-safety e riprendi (coerenza con v1.0)

- Fase 1 (trascrizione): testo salvato man mano **senza** etichette → un crash lì
  conserva comunque il trascritto.
- Le etichette si aggiungono **riscrivendo il file alla fine**. Crash durante
  "Riconoscimento voci" → testo già salvato, manca solo l'attribuzione voci.
- **Riprendi**: continua la trascrizione dal punto interrotto (come ora); le voci
  sono calcolate sull'intero audio alla fine. La diarization non è ripartibile a
  metà (il clustering è globale): se serve, si rifà la sola fase voci.

## Modelli offline e dipendenze

- Modelli pyannote 3.1 (segmentation + embedding) scaricati **una volta** su
  questo PC con un **token HuggingFace** dell'utente (account gratuito + accettare
  i termini dei modelli). Poi **impacchettati dentro l'exe** (decine di MB) e
  caricati da percorso locale a runtime, senza rete.
- Nuove dipendenze: `pyannote.audio` + `torchaudio` (versione compatibile con il
  torch già presente). L'exe cresce stimato da ~190 MB a ~250–350 MB.

## Rischio principale: pacchettizzazione (PyInstaller)

pyannote/torchaudio hanno import nascosti e file dati ostici per PyInstaller. Per
questo il **primo passo dell'implementazione è uno spike di validazione**:
caricare pyannote offline dai modelli impacchettati in un exe di prova e
diarizzare una clip a 2 voci, **prima** di costruire l'interfaccia. Se emergono
problemi grossi di packaging, si adatta subito l'approccio.

## Prestazioni (CPU)

Il riconoscimento voci aggiunge grosso modo da 0,5× a 1× la durata dell'audio su
una CPU decente, di più sul PC debole. Accettabile perché opzionale.

## Test

- Clip a 2 voci: verifica etichette corrette, abbinamento sensato.
- Caricamento modelli **offline** (rete disconnessa) dentro l'exe buildato.
- Funzione **Rinomina voci** (finestra + file).
- Regressione: con casella spenta, tutto come v1.0.

## Prerequisiti / blocchi noti

- **BLOCCO per iniziare**: token HuggingFace dell'utente + accettazione dei
  termini dei modelli pyannote (legata al suo account, non delegabile). Senza
  questo non si scaricano i modelli, quindi non si può implementare/testare.
- Lavoro su branch `v2-diarization`; il `main` (release v1.0 condivisibile) resta
  stabile finché la v2 non è verificata, poi si fa il merge e la CI ricompila.

## Fuori scope (per ora)

- Identità automatica con nomi (speaker recognition con voci pre-registrate).
- Accelerazione GPU AMD (torch-directml): da esplorare dopo.
- Pulsante di rinomina avanzato oltre la sostituzione testo.
