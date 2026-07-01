#!/usr/bin/env python3
"""
vtt2txt.py
==========
Step 0.5 of the MedYT-Train pipeline — sits between subtitle download
and transcription.

Converts WebVTT subtitle files (.vtt) into clean plain-text files (.txt)
by stripping all timing cues, metadata headers, and HTML-like tags. The
resulting text is what alignment.py will use as the reference transcript
to transfer Whisper's word-level timestamps onto.

Only videos for which both a .vtt file AND a matching audio/video file
exist are processed — this ensures we only produce text files for videos
we can actually segment later.

Output: one .txt file per video under textes_propres/<channel>/<video_id>.txt

Requirements:
    stdlib only (os, glob, re)
"""
import glob
import os
import re

# !PLACEHOLDER — root folder containing one sub-directory per YouTube channel,
# each holding the downloaded video files and their .vtt subtitle files
DATASET_PATH = "!PLACEHOLDER"

# !PLACEHOLDER — where the clean plain-text files will be written
OUTPUT_BASE_PATH = "!PLACEHOLDER"

# VTT header lines we want to drop entirely
_VTT_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE")

# matches a WebVTT cue timestamp line, e.g. "00:00:01.000 --> 00:00:04.000"
_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}[\d:\.]+\s+-->\s+\d{2}:\d{2}")

# matches inline VTT tags such as <c>, </c>, <00:00:01.000>, <v Speaker>
_TAG_RE = re.compile(r"<[^>]+>")


def vtt_to_text(vtt_path: str) -> str:
    """Strip all VTT structure from a subtitle file and return the spoken
    text as a single normalised string."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    text_blocks = []
    for line in lines:
        line = line.strip()

        if not line:
            continue
        if any(line.startswith(p) for p in _VTT_HEADER_PREFIXES):
            continue
        if _TIMESTAMP_RE.search(line):
            continue

        # remove inline tags and HTML entities
        line = _TAG_RE.sub("", line)
        line = line.replace("&nbsp;", " ").replace("&amp;", "&")
        line = line.strip()

        if line:
            text_blocks.append(line)

    # collapse into one line and normalise whitespace — alignment.py splits
    # on whitespace, so the exact line breaks do not matter here
    return " ".join(" ".join(text_blocks).split())


channels = [
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
]

for channel in channels:
    channel_path = os.path.join(DATASET_PATH, channel)
    channel_out_path = os.path.join(OUTPUT_BASE_PATH, channel)
    os.makedirs(channel_out_path, exist_ok=True)

    vtt_files = glob.glob(os.path.join(channel_path, "*.vtt"))

    for vtt_file in vtt_files:
        base_path_no_ext = vtt_file.replace(".en.vtt", "").replace(".vtt", "")
        base_name = os.path.basename(base_path_no_ext)

        # only process if the source video/audio is actually present —
        # a subtitle without a video is useless downstream
        source_file = None
        for ext in [".webm", ".mkv", ".mp4", ".wav", ".m4a", ".mp3"]:
            candidate = base_path_no_ext + ext
            if os.path.exists(candidate):
                source_file = candidate
                break

        if source_file is None:
            continue

        txt_path = os.path.join(channel_out_path, base_name + ".txt")
        if os.path.exists(txt_path):
            # already converted — safe to restart
            continue

        print(f"converting: {os.path.basename(vtt_file)}")

        text = vtt_to_text(vtt_file)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

print("done — plain-text transcripts written to", OUTPUT_BASE_PATH)
