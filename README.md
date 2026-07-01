# MedYT

**MedYT** is a large-scale medical speech corpus of 8,250 hours sourced from 68 YouTube channels across 9 medical disciplines, designed for self-supervised pretraining of speech foundation models. It is accompanied by an 11-hour manually curated evaluation benchmark (MedYT-Test) and a 603-hour aligned training corpus (MedYT-Train).

This repository provides the recipes to reconstruct both corpora from publicly available YouTube content.

> **Paper:** *Benchmarking Self-Supervised Speech Models for Medical ASR: A Diverse Spoken Sources Corpus and an Empirical Study of Pretraining Strategies* 

---

## Repository structure

```
medYT/
├── urls.json                      # metadata for all 35,290 videos (id, channel, url, duration)
├── urls.txt                       # one YouTube URL per line — input to yt-dlp
│
├── medyt_unsupervised/
│   └── vad.py                     # VAD-based segmentation pipeline (Silero VAD)
│
├── medyt_train/
│   ├── vtt2txt.py                 # step 1 — convert .vtt subtitles to clean .txt
│   ├── transcription.py           # step 2 — extract word-level timestamps with Whisper
│   ├── alignment.py               # step 3 — align reference text onto Whisper timestamps
│   └── segmentation.py            # step 4 — cut audio at aligned timestamps, write CSV
│
└── download_manual_subtitles.py   # helper — download only human-written subtitles
```

---

## Corpus overview

| Corpus | Hours | Videos | Channels | Use |
|---|---|---|---|---|
| MedYT-Unsupervised | 8,249 | 35,290 | 68 | Self-supervised pretraining |
| MedYT-Train | 603 | 3,972 | 30 | Supervised ASR fine-tuning |
| MedYT-Test | 11 | — | — | Evaluation benchmark |

---

## Requirements

```bash
pip install yt-dlp openai-whisper soundfile tqdm torch
```

ffmpeg must be available on your system (`apt install ffmpeg` or equivalent).

---

## Step 1 — Download the videos

`urls.txt` contains one YouTube URL per line for all 35,290 videos in MedYT-Unsupervised. Download them as FLAC audio files with:

```bash
yt-dlp --batch-file urls.txt \
       -x --audio-format flac \
       -o "%(channel)s/%(id)s/%(id)s.%(ext)s"
```

`urls.json` provides richer metadata (channel name, number of segments, total duration) if you want to filter by category before downloading.

> **Note:** Some videos may have been deleted or made private since the corpus was assembled. This is expected for large YouTube-based datasets — `yt-dlp` will skip unavailable videos and continue.

---

## Step 2A — Reconstruct MedYT-Unsupervised

Run the VAD-based segmentation pipeline on the downloaded audio:

```bash
# set INPUT_DIR and OUTPUT_DIR inside the script first
python medyt_unsupervised/vad.py
```

This uses [Silero VAD](https://github.com/snakers4/silero-vad) to detect speech activity and merges consecutive chunks into segments of 2–30 seconds. The script is parallelised across GPUs — set `num_workers` to match your setup.

---

## Step 2B — Reconstruct MedYT-Train

MedYT-Train requires human-written subtitles. Run the four steps in order:

### Download human-written subtitles

```bash
python download_manual_subtitles.py \
    --urls urls.txt \
    --out-dir subtitles/ \
    --lang en \
    --workers 8 \
    --log subtitle_download.log
```

Only videos with manually written subtitle tracks are downloaded — auto-generated YouTube captions are detected and skipped automatically.

### Run the alignment pipeline

```bash
# 1. convert .vtt files to clean plain text
python medyt_train/vtt2txt.py

# 2. extract word-level timestamps with Whisper (text is discarded, only timing is kept)
python medyt_train/transcription.py

# 3. align the clean reference text onto the Whisper timestamps
python medyt_train/alignment.py

# 4. cut the audio at aligned timestamps and write the metadata CSV
python medyt_train/segmentation.py
```

Set the `!PLACEHOLDER` paths at the top of each script before running. Each script is safe to restart — already-processed files are skipped.

---

## MedYT-Test

MedYT-Test is a manually curated 11-hour evaluation benchmark structured along two axes: communicative genre and clinical domain. It is distributed separately as pre-segmented audio with reference transcriptions:

> [!PLACEHOLDER — download link]

---

## License

The audio and transcripts are derived from publicly available YouTube content. Each channel's content is subject to its original creator's license. The recipes, code, and metadata in this repository are released under [!PLACEHOLDER — e.g. MIT / CC-BY 4.0].


