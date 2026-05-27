# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Whisper carica a runtime mel_filters.npz e i tokenizer (*.tiktoken) dalla
# sua cartella assets/: vanno raccolti come datas, altrimenti la trascrizione
# fallisce con FileNotFoundError dentro l'exe.
datas = collect_data_files('whisper')
hiddenimports = collect_submodules('whisper') + [
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
]

a = Analysis(
    ['sbobinator.py'],
    pathex=[],
    binaries=[],
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
    name='Sbobinator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)
