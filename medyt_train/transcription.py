#!/usr/bin/env python3
"""
transcription.py
================
Step 1 of the MedYT-Train pipeline.

Runs Whisper on each video that has a matching human-written subtitle file
(.vtt) and extracts word-level timestamps. The Whisper transcript itself is
discarded — only the timing information is kept as a temporal pivot for the
alignment step that follows.

Output: one JSON file per video under asr_pivot/<channel>/<video_id>_asr.json
        Each file is a list of {"word": ..., "start": ..., "end": ...} entries.

Requirements:
    pip install openai-whisper
"""
import os
import glob
import json

import whisper

# !PLACEHOLDER — root folder containing one sub-directory per YouTube channel,
# each holding the downloaded video files and their .vtt subtitle files
DATASET_PATH = "!PLACEHOLDER"

# !PLACEHOLDER — where the word-level timestamp JSONs will be written
ASR_OUT_PATH = "!PLACEHOLDER"

# !PLACEHOLDER — local path where Whisper model weights are cached
# (useful on compute nodes without internet access)
MODELS_PATH = "!PLACEHOLDER"

# Whisper model size — "small" is a good trade-off between speed and alignment
# quality when the transcript is only used as a timing pivot
WHISPER_MODEL = "small"

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(ASR_OUT_PATH, exist_ok=True)

print(f"Loading Whisper model ({WHISPER_MODEL})...")
model = whisper.load_model(WHISPER_MODEL, download_root=MODELS_PATH)

channels = [
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
]

for channel in channels:
    channel_path = os.path.join(DATASET_PATH, channel)
    channel_out = os.path.join(ASR_OUT_PATH, channel)
    os.makedirs(channel_out, exist_ok=True)

    # we only process videos for which a human-written subtitle file exists —
    # the .vtt is what determines whether a video ends up in MedYT-Train
    vtt_files = glob.glob(os.path.join(channel_path, "*.vtt"))

    for vtt_file in vtt_files:
        base_path_no_ext = vtt_file.replace(".en.vtt", "").replace(".vtt", "")
        base_name = os.path.basename(base_path_no_ext)

        # find the actual video file — yt-dlp can produce several container formats
        video_path = None
        for ext in [".webm", ".mkv", ".mp4", ".wav", ".m4a"]:
            candidate = base_path_no_ext + ext
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            print(f"warning: no video file found for subtitle '{os.path.basename(vtt_file)}' — skipping")
            continue

        output_json = os.path.join(channel_out, base_name + "_asr.json")
        if os.path.exists(output_json):
            # already processed — safe to restart the job mid-way
            continue

        print(f"extracting timestamps: {os.path.basename(video_path)}")

        # word_timestamps=True is what makes Whisper useful here — without it
        # we only get segment-level timings which are too coarse for alignment
        result = model.transcribe(video_path, word_timestamps=True, language="en")

        words_data = []
        for segment in result["segments"]:
            for word in segment["words"]:
                words_data.append({
                    "word": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"],
                })

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(words_data, f, ensure_ascii=False, indent=2)

print("done — word-level timestamps written to", ASR_OUT_PATH)
