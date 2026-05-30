# Spec "dev fast": onedir invece di onefile = build molto piu' rapida.
# L'app diventa una cartella con un .exe + librerie, ma per iterare in dev
# e' enormemente piu' veloce.
# Uso: pyinstaller sbobinator-dev.spec --noconfirm --clean --distpath dist_dev --workpath build_dev
#
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

datas = collect_data_files('whisper') + [('hf_models', 'hf_models')]
binaries = []
hiddenimports = collect_submodules('whisper') + [
    'tiktoken', 'tiktoken_ext', 'tiktoken_ext.openai_public',
]

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
    noarchive=True,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='verbaLIA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='verbaLIA',
)
