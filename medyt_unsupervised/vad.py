import os
import glob
import torch
import subprocess
import multiprocessing as mp
from pathlib import Path
from uuid import uuid4

# ================= CONFIGURATION =================
INPUT_DIR = "!PLACEHOLDER"   # root directory containing the raw downloaded videos
OUTPUT_DIR = "!PLACEHOLDER"  # where the segmented FLAC files will be written
SAMPLE_RATE = 16000           # target sample rate (Hz) — keep at 16kHz for ASR
MIN_SEC = 2                   # discard segments shorter than this (seconds)
MAX_SEC = 30                  # hard cap on segment length (seconds)
# =================================================

# these are shared across worker processes and set during initialisation
worker_device = None
worker_model = None
worker_utils = None


def init_worker(gpu_queue):
    """Initialise one VAD worker: pull a GPU id from the shared queue and
    load Silero VAD onto that device. Using a queue ensures workers spread
    evenly across available GPUs without any two workers fighting over the
    same one."""
    global worker_device, worker_model, worker_utils
    torch.set_num_threads(1)

    gpu_id = gpu_queue.get()
    worker_device = torch.device(f"cuda:{gpu_id}")

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    worker_model = model.to(worker_device)
    worker_utils = utils


def process_file(video_path):
    """Segment a single video file using Silero VAD and save the resulting
    chunks as FLAC files under OUTPUT_DIR, mirroring the source directory
    structure. Already-processed videos are skipped so the job can be safely
    restarted after a failure."""
    global worker_device, worker_model, worker_utils
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = worker_utils

    video_name = Path(video_path).stem

    # mirror the source folder hierarchy so it is easy to trace a segment
    # back to its original video later
    rel_path = os.path.relpath(video_path, INPUT_DIR)
    rel_dir = os.path.dirname(rel_path)
    out_subfolder = os.path.join(OUTPUT_DIR, rel_dir, video_name)

    if os.path.exists(out_subfolder) and len(os.listdir(out_subfolder)) > 0:
        return f"already done: {rel_dir}/{video_name}"

    os.makedirs(out_subfolder, exist_ok=True)

    # decode to a temporary WAV on local scratch — much faster I/O than
    # writing intermediate files to the network filesystem
    temp_dir = os.environ.get("JOBSCRATCH", "/tmp")
    temp_wav = os.path.join(temp_dir, f"{video_name}_{uuid4().hex[:6]}.wav")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                temp_wav,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        wav = read_audio(temp_wav, sampling_rate=SAMPLE_RATE).to(worker_device)

        speech_timestamps = get_speech_timestamps(
            wav,
            worker_model,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )

        min_samples = MIN_SEC * SAMPLE_RATE
        max_samples = MAX_SEC * SAMPLE_RATE

        # greedily merge consecutive speech chunks until we hit the MAX_SEC
        # ceiling — this keeps segments long enough to be useful for
        # self-supervised pretraining while avoiding very long sequences
        valid_segments = []
        current_start = None
        current_end = None

        for ts in speech_timestamps:
            start = ts["start"]
            end = ts["end"]
            if current_start is None:
                current_start = start
                current_end = end
                continue
            if (end - current_start) <= max_samples:
                current_end = end
            else:
                if (current_end - current_start) >= min_samples:
                    valid_segments.append((current_start, current_end))
                current_start = start
                current_end = end

        if current_start is not None and (current_end - current_start) >= min_samples:
            valid_segments.append((current_start, current_end))

        for i, (s_start, s_end) in enumerate(valid_segments):
            chunk = wav[s_start:s_end].cpu()
            out_path = os.path.join(out_subfolder, f"{video_name}_{i + 1:04d}.flac")
            save_audio(out_path, chunk, SAMPLE_RATE)

    except Exception as e:
        return f"error — {video_name}: {e}"
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    return f"ok: [{rel_dir}] {video_name} ({len(valid_segments)} segments)"


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extensions = ["*.mkv", "*.webm", "*.mp4", "*.avi", "*.mov", "*.m4a"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, "**", ext), recursive=True))

    num_gpus = torch.cuda.device_count()
    num_workers = 32  # !PLACEHOLDER — tune to your cluster (typically 4–8× num_gpus)

    m = mp.Manager()
    gpu_queue = m.Queue()
    for i in range(num_workers):
        gpu_queue.put(i % num_gpus)

    total_files = len(files)
    processed = 0

    print(f"found {total_files} video files — starting {num_workers} workers across {num_gpus} GPU(s)")

    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(gpu_queue,)) as pool:
        for result in pool.imap_unordered(process_file, files):
            processed += 1
            print(f"[{processed}/{total_files}] {result}", flush=True)