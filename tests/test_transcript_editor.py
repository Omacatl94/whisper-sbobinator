import pytest

import transcript_editor as te


@pytest.fixture(scope="module")
def tk_root():
    """Un unico root Tk per tutto il modulo: creare più Tk() nello stesso
    processo fa fallire l'init di Tcl (init.tcl). L'app reale ha un solo root."""
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


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


def test_text_with_special_chars_roundtrip():
    text = "[00:10 - 00:14] però: [nota] caffè\n"
    assert te.serialize_segments(te.parse_segments(text)) == text


def test_serialize_uses_edited_text_and_keeps_timestamps():
    segs = [{"start": 65.0, "end": 69.0, "text": "testo corretto"}]
    assert te.serialize_segments(segs) == "[01:05 - 01:09] testo corretto\n"


def test_play_segment_slices_and_plays(monkeypatch):
    import sys
    import numpy as np
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

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD)
    te.play_segment(audio, 2.0, 5.0)
    assert played["stopped"] is True
    assert played["sr"] == 16000
    assert played["len"] == 16000 * 3   # 3 secondi


def test_open_editor_builds_without_error(tk_root, tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("[00:00 - 00:04] uno\n[00:04 - 00:08] due\n", encoding="utf-8")
    win = te.open_editor(tk_root, str(tmp_path / "a.wav"), str(txt))
    win.withdraw()
    tk_root.update_idletasks()
    tk_root.update()
    try:
        assert win.winfo_exists()
    finally:
        win.destroy()


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def test_editor_save_rewrites_txt_and_calls_on_saved(tk_root, tmp_path, monkeypatch):
    import tkinter as tk
    monkeypatch.setattr(te.messagebox, "showinfo", lambda *a, **k: None)
    txt = tmp_path / "a.txt"
    txt.write_text("[00:00 - 00:04] vecchio\n", encoding="utf-8")
    saved = {}
    win = te.open_editor(tk_root, str(tmp_path / "a.wav"), str(txt),
                         on_saved=lambda p: saved.update(path=p))
    win.withdraw()
    tk_root.update_idletasks()
    tk_root.update()
    try:
        entry = next(w for w in _walk(win) if isinstance(w, tk.Entry))
        entry.delete(0, "end")
        entry.insert(0, "nuovo testo")
        save_btn = next(w for w in _walk(win)
                        if isinstance(w, tk.Button) and w.cget("text") == "Salva")
        save_btn.invoke()
        assert saved.get("path") == str(txt)
        assert txt.read_text(encoding="utf-8") == "[00:00 - 00:04] nuovo testo\n"
    finally:
        win.destroy()
