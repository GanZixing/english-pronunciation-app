import os
import shutil
import subprocess


def allowed_ext(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in [".wav", ".webm", ".mp3", ".m4a", ".ogg", ".flac"]


def convert_to_wav(input_path: str) -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it is in PATH.")

    output_path = os.path.splitext(input_path)[0] + ".wav"

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {result.stderr}")

    if not os.path.exists(output_path):
        raise RuntimeError("wav file not generated after ffmpeg conversion")

    return output_path
