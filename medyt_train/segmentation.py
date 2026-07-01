#!/usr/bin/env python3
"""
segmentation.py
===============
Step 3 (final) of the MedYT-Train pipeline.

Reads the aligned JSONL files produced by alignment.py and cuts the original
video files into individual WAV segments at the timestamps provided. Each
segment is saved as a separate file and logged in a global metadata CSV.

The heavy lifting (decoding the full video to PCM) is done once per video
using ffmpeg, and the resulting waveform is held in RAM while all segments
for that video are sliced out — much faster than calling ffmpeg once per
segment.

Output:
    segments_audio/<channel>/<video_id>_<NNNN>.wav  — one file per segment
    metadata.csv                                     — global index

Requirements:
    pip install soundfile
"""
import csv
import glob
import json
import os
import subprocess

import soundfile as sf

# !PLACEHOLDER — folder containing the original downloaded video files,
# organised as <media_base_path>/<channel>/<video_id>.<ext>
MEDIA_BASE_PATH = "!PLACEHOLDER"

# !PLACEHOLDER — output of alignment.py (JSONL files with start/end/text)
JSONL_BASE_PATH = "!PLACEHOLDER"

# !PLACEHOLDER — where the segmented WAV files will be written
OUTPUT_AUDIO_PATH = "!PLACEHOLDER"

# !PLACEHOLDER — path for the global metadata CSV
CSV_OUTPUT_FILE = "!PLACEHOLDER"

# use the node's local scratch disk for the temporary full-video WAV —
# much faster than writing to the network filesystem
TMP_DIR = os.environ.get("TMPDIR", "/tmp")

SAMPLE_RATE = 16000  # all segments are resampled to 16 kHz mono

os.makedirs(OUTPUT_AUDIO_PATH, exist_ok=True)

csv_data = []

channels = [
    d for d in os.listdir(JSONL_BASE_PATH)
    if os.path.isdir(os.path.join(JSONL_BASE_PATH, d))
]

for channel in channels:
    channel_jsonl_dir = os.path.join(JSONL_BASE_PATH, channel)
    channel_media_dir = os.path.join(MEDIA_BASE_PATH, channel)
    channel_out_dir = os.path.join(OUTPUT_AUDIO_PATH, channel)
    os.makedirs(channel_out_dir, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(channel_jsonl_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        base_name = os.path.splitext(os.path.basename(jsonl_file))[0]

        # locate the source video — yt-dlp can leave files in several formats
        video_file = None
        for ext in [".webm", ".mkv", ".mp4", ".wav", ".m4a"]:
            candidate = os.path.join(channel_media_dir, base_name + ext)
            if os.path.exists(candidate):
                video_file = candidate
                break

        if video_file is None:
            print(f"warning: source video not found for {base_name}.jsonl — skipping")
            continue

        print(f"segmenting: {base_name}")

        # decode the whole video to a temporary WAV once, then slice it in RAM
        # — avoids spawning one ffmpeg process per segment
        tmp_wav = os.path.join(TMP_DIR, f"{base_name}_full.wav")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_file,
                    "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le",
                    "-v", "quiet",
                    tmp_wav,
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"error: ffmpeg failed on {video_file}")
            continue

        try:
            audio_data, sample_rate = sf.read(tmp_wav)
        except Exception as e:
            print(f"error: could not read audio — {e}")
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
            continue

        with open(jsonl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            segment_info = json.loads(line)
            start_sec = segment_info["start"]
            end_sec = segment_info["end"]
            text = segment_info["text"]

            duration = round(end_sec - start_sec, 3)
            segment_id = f"{base_name}_{i:04d}"
            segment_path = os.path.join(channel_out_dir, f"{segment_id}.wav")

            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            sf.write(segment_path, segment_audio, sample_rate)

            csv_data.append({
                "id": segment_id,
                "audio_path": segment_path,
                "duration": duration,
                "text": text,
            })

        # clean up the temporary full-video WAV to avoid filling the scratch disk
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

# write the global metadata CSV — this becomes the index for SpeechBrain / HF datasets
print(f"\nwriting metadata CSV: {CSV_OUTPUT_FILE}")
with open(CSV_OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["id", "audio_path", "duration", "text"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)

print("done — MedYT-Train dataset created successfully")
