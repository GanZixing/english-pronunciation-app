import os
import uuid

from flask import Flask, request, jsonify

from services.transcriber import transcribe_with_words
from services.text_compare import align_target_and_recognized
from services.audio_features import extract_sentence_rhythm, extract_word_features
from services.suspicious_words import detect_suspicious_words
from utils.text_normalize import normalize_text
from utils.audio_convert import allowed_ext, convert_to_wav
from services.phonetics import get_sentence_phonetics

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def build_simple_feedback(target_text: str, recognized_text: str, suspicious_words: list) -> list:
    feedback = []

    norm_target = normalize_text(target_text)
    norm_rec = normalize_text(recognized_text)

    if norm_target != norm_rec:
        feedback.append("The recognized sentence does not fully match your target sentence.")

    if suspicious_words:
        top_words = ", ".join([w["word"] for w in suspicious_words[:3] if w["word"]])
        if top_words:
            feedback.append(f"These words may need attention: {top_words}")

    if not feedback:
        feedback.append("Overall, the sentence looks fairly stable from the current analysis.")

    return feedback


@app.route("/", methods=["GET"])
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI Pronunciation Trainer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 30px auto;
            line-height: 1.5;
        }
        input, button {
            font-size: 16px;
        }
        #targetText {
            width: 100%;
            padding: 8px;
        }
        button {
            margin-right: 10px;
            padding: 10px 16px;
        }
        #status {
            margin-top: 12px;
            font-weight: bold;
        }
        #result {
            margin-top: 20px;
            white-space: pre-wrap;
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
        }
        .hint {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h2>AI Pronunciation Trainer</h2>

    <p>Target sentence:</p>
    <input id="targetText" value="I have three dogs.">

    <p class="hint">Click Start Recording, read the sentence, then click Stop Recording.</p>

    <button id="startBtn">Start Recording</button>
    <button id="stopBtn" disabled>Stop Recording</button>

    <p id="status">Idle</p>

    <audio id="audioPlayer" controls style="width:100%; margin-top:10px;"></audio>

    <div id="prettyResult" style="margin-top:20px;"></div>
    <pre id="result"></pre>

<script>
let mediaRecorder = null;
let audioChunks = [];
let currentStream = null;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const prettyResultEl = document.getElementById("prettyResult");
const audioPlayer = document.getElementById("audioPlayer");

function escapeHtml(text) {
    return text.replace(/[&<>"]/g, function(m) {
        return ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;'
        })[m];
    });
}

async function startRecording() {
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(currentStream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        mediaRecorder.onstart = () => {
            statusEl.innerText = "Recording...";
            resultEl.textContent = "";
            prettyResultEl.innerHTML = "";
            startBtn.disabled = true;
            stopBtn.disabled = false;
        };

        mediaRecorder.start();
    } catch (err) {
        statusEl.innerText = "Microphone error: " + err.message;
    }
}

async function stopRecording() {
    if (!mediaRecorder) return;

    mediaRecorder.onstop = async () => {
        try {
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            audioPlayer.src = URL.createObjectURL(blob);

            const formData = new FormData();
            formData.append("audio_file", blob, "recording.webm");
            formData.append("target_text", document.getElementById("targetText").value);

            statusEl.innerText = "Uploading and analyzing...";

            const response = await fetch("/analyze", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            resultEl.textContent = JSON.stringify(data, null, 2);

            if (!response.ok) {
                prettyResultEl.innerHTML = "<p style='color:red;'>Analyze failed.</p>";
                statusEl.innerText = "Failed";
                return;
            }

            let suspiciousHtml = "";
            if (data.suspicious_words && data.suspicious_words.length > 0) {
                suspiciousHtml = "<ul>" + data.suspicious_words.slice(0, 5).map(w => {
                    return "<li><b>" + escapeHtml(w.word) + "</b> (score: " + w.score + ") - "
                        + escapeHtml((w.reasons || []).join("; ")) + "</li>";
                }).join("") + "</ul>";
            } else {
                suspiciousHtml = "<p>No obvious suspicious words detected.</p>";
            }

            let feedbackHtml = "";
            if (data.feedback && data.feedback.length > 0) {
                feedbackHtml = "<ul>" + data.feedback.map(x => "<li>" + escapeHtml(x) + "</li>").join("") + "</ul>";
            }

            let phoneticHtml = "";
            if (data.phonetics) {
                phoneticHtml = `
                    <h3>Phonetics</h3>
                    <p><b>IPA:</b> /${escapeHtml(data.phonetics.ipa_sentence || "")}/</p>
                    <p><b>ARPAbet:</b> ${escapeHtml(data.phonetics.arpabet_sentence || "")}</p>
                `;
            }

            prettyResultEl.innerHTML = `
                <h3>Summary</h3>
                <p><b>Target:</b> ${escapeHtml(data.target_text || "")}</p>
                <p><b>Recognized:</b> ${escapeHtml(data.recognized_text || "")}</p>
                <p><b>Similarity:</b> ${data.similarity_score ?? ""}</p>

                ${phoneticHtml}

                <h3>Feedback</h3>
                ${feedbackHtml}

                <h3>Suspicious Words</h3>
                ${suspiciousHtml}
            `;

            statusEl.innerText = "Done";
        } catch (err) {
            statusEl.innerText = "Error: " + err.message;
        } finally {
            startBtn.disabled = false;
            stopBtn.disabled = true;

            if (currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
                currentStream = null;
            }
        }
    };

    mediaRecorder.stop();
}

startBtn.onclick = startRecording;
stopBtn.onclick = stopRecording;
</script>
</body>
</html>
"""


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        target_text = request.form.get("target_text", "").strip()
        audio_file = request.files.get("audio_file")

        if not target_text:
            return jsonify({"error": "target_text is required"}), 400

        if not audio_file:
            return jsonify({"error": "audio_file is required"}), 400

        if not allowed_ext(audio_file.filename):
            return jsonify({"error": "unsupported audio format"}), 400

        ext = os.path.splitext(audio_file.filename)[1].lower() or ".webm"
        raw_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
        audio_file.save(raw_path)

        wav_path = convert_to_wav(raw_path)

        transcribed = transcribe_with_words(wav_path, language="en")
        alignment = align_target_and_recognized(target_text, transcribed["recognized_text"])
        rhythm = extract_sentence_rhythm(transcribed["words"], transcribed.get("duration"))
        word_features = extract_word_features(wav_path, transcribed["words"])
        suspicious_words = detect_suspicious_words(word_features)

        normalized_target = normalize_text(target_text)
        normalized_recognized = normalize_text(transcribed["recognized_text"])

        target_set = set(normalized_target.split())
        rec_set = set(normalized_recognized.split())
        similarity = len(target_set & rec_set) / max(len(target_set), 1)

        phonetics = get_sentence_phonetics(target_text)
        feedback = build_simple_feedback(
            target_text=target_text,
            recognized_text=transcribed["recognized_text"],
            suspicious_words=suspicious_words
        )

        return jsonify({
            "target_text": target_text,
            "recognized_text": transcribed["recognized_text"],
            "normalized_target": normalized_target,
            "normalized_recognized": normalized_recognized,
            "similarity_score": round(float(similarity), 4),
            "transcription": {
                "language": transcribed.get("language"),
                "duration": transcribed.get("duration"),
                "words": transcribed["words"]
            },
            "alignment": alignment,
            "rhythm": rhythm,
            "word_features": word_features,
            "suspicious_words": suspicious_words,
            "feedback": feedback,
            "phonetics": phonetics
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)