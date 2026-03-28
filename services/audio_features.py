import numpy as np
import librosa


def load_audio(audio_path: str, sr: int = 16000):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr


def safe_slice(y, sr, start, end):
    start_idx = max(0, int(start * sr))
    end_idx = min(len(y), int(end * sr))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    return y[start_idx:end_idx]


def extract_sentence_rhythm(words: list, audio_duration: float | None) -> dict:
    if not words:
        return {
            "total_duration": float(audio_duration or 0.0),
            "words_per_second": 0.0,
            "avg_word_duration": 0.0,
            "avg_pause_duration": 0.0,
            "max_pause_duration": 0.0,
            "word_durations": [],
            "pauses": []
        }

    word_durations = [max(0.0, w["end"] - w["start"]) for w in words]
    pauses = []

    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        pauses.append(max(0.0, gap))

    if audio_duration is None:
        total_duration = max(1e-6, words[-1]["end"] - words[0]["start"])
    else:
        total_duration = max(1e-6, float(audio_duration))

    return {
        "total_duration": float(total_duration),
        "words_per_second": float(len(words) / total_duration),
        "avg_word_duration": float(np.mean(word_durations)) if word_durations else 0.0,
        "avg_pause_duration": float(np.mean(pauses)) if pauses else 0.0,
        "max_pause_duration": float(np.max(pauses)) if pauses else 0.0,
        "word_durations": [float(x) for x in word_durations],
        "pauses": [float(x) for x in pauses]
    }


def extract_word_features(audio_path: str, words: list, sr: int = 16000) -> list:
    y, sr = load_audio(audio_path, sr=sr)
    results = []

    for w in words:
        seg = safe_slice(y, sr, w["start"], w["end"])
        duration = max(0.0, w["end"] - w["start"])

        base = {
            "word": w["word"],
            "start": float(w["start"]),
            "end": float(w["end"]),
            "probability": float(w.get("probability", 0.0)),
            "duration": float(duration)
        }

        if len(seg) == 0:
            results.append({
                **base,
                "rms_mean": 0.0,
                "rms_max": 0.0,
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "zcr_mean": 0.0,
                "spectral_centroid_mean": 0.0,
                "speech_ratio": 0.0
            })
            continue

        rms = librosa.feature.rms(y=seg)[0]
        zcr = librosa.feature.zero_crossing_rate(seg)[0]
        centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)[0]

        try:
            f0, voiced_flag, _ = librosa.pyin(
                seg,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr
            )

            valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            speech_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
            f0_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
            f0_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0
        except Exception:
            speech_ratio = 0.0
            f0_mean = 0.0
            f0_std = 0.0

        results.append({
            **base,
            "rms_mean": float(np.mean(rms)) if len(rms) > 0 else 0.0,
            "rms_max": float(np.max(rms)) if len(rms) > 0 else 0.0,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "zcr_mean": float(np.mean(zcr)) if len(zcr) > 0 else 0.0,
            "spectral_centroid_mean": float(np.mean(centroid)) if len(centroid) > 0 else 0.0,
            "speech_ratio": speech_ratio
        })

    return results