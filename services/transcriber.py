from faster_whisper import WhisperModel

_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )
    return _whisper_model


def transcribe_with_words(audio_path: str, language: str = "en") -> dict:
    model = get_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True
    )

    full_text = []
    words = []

    for segment in segments:
        seg_text = (segment.text or "").strip()
        if seg_text:
            full_text.append(seg_text)

        if getattr(segment, "words", None):
            for w in segment.words:
                words.append({
                    "word": (w.word or "").strip(),
                    "start": float(w.start or 0.0),
                    "end": float(w.end or 0.0),
                    "probability": float(w.probability or 0.0)
                })

    return {
        "recognized_text": " ".join(full_text).strip(),
        "language": getattr(info, "language", language),
        "duration": getattr(info, "duration", None),
        "words": words
    }