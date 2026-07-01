#!/usr/bin/env python3
"""
alignment.py
============
Step 2 of the MedYT-Train pipeline.

Aligns the clean human-written transcript (from the .vtt subtitle file,
converted to plain text) with the word-level timestamps produced by Whisper
in the previous step. The alignment is done with Python's difflib, which finds
the longest common subsequences between the two word sequences after
normalising punctuation and case.

The result is a set of sentence-level segments where each segment carries:
  - the clean reference text (human-written, punctuated)
  - accurate start/end timestamps inherited from the Whisper pivot

Output: one JSONL file per video under aligned_dataset/<channel>/<video_id>.jsonl
        Each line: {"start": float, "end": float, "text": str}

Requirements:
    stdlib only (os, glob, json, difflib, re)
"""
import difflib
import glob
import json
import os
import re

# !PLACEHOLDER — folder containing one sub-directory per channel,
# each holding plain-text versions of the human subtitle files (.txt)
TXT_BASE_PATH = "!PLACEHOLDER"

# !PLACEHOLDER — output of transcription.py (word-level timestamp JSONs)
ASR_BASE_PATH = "!PLACEHOLDER"

# !PLACEHOLDER — where the aligned JSONL files will be written
OUT_BASE_PATH = "!PLACEHOLDER"

# minimum segment duration in seconds — shorter segments are discarded
# because they are unlikely to carry enough context for ASR training
MIN_DURATION_SEC = 0.5

os.makedirs(OUT_BASE_PATH, exist_ok=True)

channels = [
    d for d in os.listdir(TXT_BASE_PATH)
    if os.path.isdir(os.path.join(TXT_BASE_PATH, d))
]

for channel in channels:
    channel_txt_path = os.path.join(TXT_BASE_PATH, channel)
    channel_asr_path = os.path.join(ASR_BASE_PATH, channel)
    channel_out_path = os.path.join(OUT_BASE_PATH, channel)
    os.makedirs(channel_out_path, exist_ok=True)

    txt_files = glob.glob(os.path.join(channel_txt_path, "*.txt"))

    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        asr_file = os.path.join(channel_asr_path, base_name + "_asr.json")

        if not os.path.exists(asr_file):
            # Whisper pivot missing — transcription.py probably skipped this video
            continue

        print(f"aligning: {base_name}")

        with open(txt_file, "r", encoding="utf-8") as f:
            clean_text = f.read()

        with open(asr_file, "r", encoding="utf-8") as f:
            asr_data = json.load(f)

        # normalise both sequences the same way so difflib can match them
        # regardless of capitalisation or punctuation differences
        clean_words_raw = clean_text.split()
        clean_norm = [re.sub(r"\W+", "", w.lower()) for w in clean_words_raw]
        asr_norm = [re.sub(r"\W+", "", item["word"].lower()) for item in asr_data]

        matcher = difflib.SequenceMatcher(None, clean_norm, asr_norm)

        # build a lookup table: clean word index -> (start_sec, end_sec)
        # only matched words get a timestamp — unmatched ones are skipped
        clean_to_asr_time = {}
        for match in matcher.get_matching_blocks():
            for i in range(match.size):
                clean_idx = match.a + i
                asr_idx = match.b + i
                clean_to_asr_time[clean_idx] = (
                    asr_data[asr_idx]["start"],
                    asr_data[asr_idx]["end"],
                )

        # reconstruct sentence-level segments by splitting on strong punctuation
        # (.  ?  !) — this mirrors the segmentation described in the paper
        segments = []
        current_phrase = []
        start_time = None
        end_time = None

        for i, word in enumerate(clean_words_raw):
            current_phrase.append(word)

            if i in clean_to_asr_time:
                if start_time is None:
                    start_time = clean_to_asr_time[i][0]
                end_time = clean_to_asr_time[i][1]

            if word.endswith((".", "?", "!")) and start_time is not None:
                duration = (end_time - start_time) if end_time is not None else 0
                if duration > MIN_DURATION_SEC:
                    segments.append({
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "text": " ".join(current_phrase),
                    })
                # reset for the next sentence
                current_phrase = []
                start_time = None
                end_time = None

        out_file = os.path.join(channel_out_path, base_name + ".jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

print("done — aligned segments written to", OUT_BASE_PATH)
