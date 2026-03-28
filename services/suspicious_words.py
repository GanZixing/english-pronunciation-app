import numpy as np


def detect_suspicious_words(word_features: list) -> list:
    if not word_features:
        return []

    durations = [w["duration"] for w in word_features if w["duration"] > 0]
    rms_values = [w["rms_mean"] for w in word_features]

    avg_duration = float(np.mean(durations)) if durations else 0.0
    avg_rms = float(np.mean(rms_values)) if rms_values else 0.0

    suspicious = []

    for idx, w in enumerate(word_features):
        reasons = []
        score = 0.0

        prob = w.get("probability", 1.0)
        duration = w.get("duration", 0.0)
        rms_mean = w.get("rms_mean", 0.0)
        speech_ratio = w.get("speech_ratio", 0.0)

        if prob < 0.75:
            reasons.append(f"ASR confidence low ({prob:.2f})")
            score += 0.35

        if avg_duration > 0 and duration > avg_duration * 1.8:
            reasons.append(f"word duration too long ({duration:.2f}s)")
            score += 0.20
        elif avg_duration > 0 and duration < avg_duration * 0.45:
            reasons.append(f"word duration too short ({duration:.2f}s)")
            score += 0.20

        if avg_rms > 0 and rms_mean < avg_rms * 0.45:
            reasons.append("energy too weak")
            score += 0.20

        if speech_ratio < 0.35:
            reasons.append(f"low voiced ratio ({speech_ratio:.2f})")
            score += 0.15

        if reasons:
            suspicious.append({
                "index": idx,
                "word": w["word"],
                "start": float(w["start"]),
                "end": float(w["end"]),
                "score": round(min(score, 1.0), 3),
                "reasons": reasons
            })

    suspicious.sort(key=lambda x: x["score"], reverse=True)
    return suspicious