# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

# Whisper carica a runtime mel_filters.npz e i tokenizer (*.tiktoken) dalla sua
# cartella assets/: vanno raccolti come datas.
# hf_models = modelli pyannote bundled per diarization offline.
datas = collect_data_files('whisper') + [('hf_models', 'hf_models')]
binaries = []
hiddenimports = collect_submodules('whisper') + [
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
    'transcript_editor',
]

# pyannote.audio e dipendenze: raccogli dati, binari e import nascosti.
# I pacchetti non installati vengono semplicemente ignorati.
for pkg in [
    'pyannote.audio', 'pyannote.core', 'pyannote.pipeline',
    'pyannote.database', 'pyannote.metrics',
    'torchaudio', 'torchcodec',
    'asteroid_filterbanks', 'torch_audiomentations',
    'pytorch_lightning', 'lightning_fabric', 'lightning',
    'speechbrain', 'huggingface_hub',
    'soundfile', 'sounddevice',
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

a = Analysis(
    ['sbobinator.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='verbaLIA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)
