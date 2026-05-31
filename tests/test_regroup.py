import sbobinator as s


def test_regroup_splits_on_sentence_end():
    result = {"segments": [{"words": [
        {"start": 0.0, "end": 0.5, "word": " Ciao"},
        {"start": 0.5, "end": 1.0, "word": " mondo."},
        {"start": 1.2, "end": 1.6, "word": " Come"},
        {"start": 1.6, "end": 2.0, "word": " stai?"},
    ]}]}
    segs = s.regroup_sentences(result)
    assert len(segs) == 2
    assert segs[0] == {"start": 0.0, "end": 1.0, "text": "Ciao mondo."}
    assert segs[1] == {"start": 1.2, "end": 2.0, "text": "Come stai?"}


def test_regroup_splits_on_long_pause():
    result = {"segments": [{"words": [
        {"start": 0.0, "end": 0.4, "word": " uno"},
        {"start": 0.4, "end": 0.8, "word": " due"},
        {"start": 2.8, "end": 3.2, "word": " tre"},       # pausa di 2 s
        {"start": 3.2, "end": 3.6, "word": " quattro"},
    ]}]}
    segs = s.regroup_sentences(result, max_gap=0.8)
    assert len(segs) == 2
    assert segs[0] == {"start": 0.0, "end": 0.8, "text": "uno due"}
    assert segs[1] == {"start": 2.8, "end": 3.6, "text": "tre quattro"}


def test_regroup_fallback_without_words():
    result = {"segments": [{"start": 0.0, "end": 2.0, "text": "frase senza words"}]}
    assert s.regroup_sentences(result) == \
        [{"start": 0.0, "end": 2.0, "text": "frase senza words"}]


def test_regroup_splits_on_max_chars_when_no_punctuation():
    words = [{"start": float(i), "end": float(i) + 0.4, "word": " parola"}
             for i in range(60)]
    result = {"segments": [{"words": words}]}
    segs = s.regroup_sentences(result, max_chars=50)
    assert len(segs) > 1
    assert all(len(x["text"]) <= 60 for x in segs)
