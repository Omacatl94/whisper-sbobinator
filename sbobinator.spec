# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['sbobinator.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['whisper', 'whisper.assets', 'tiktoken', 'tiktoken_ext', 'tiktoken_ext.openai_public'],
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
