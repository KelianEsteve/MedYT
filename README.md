# MedYT
Recipe for downloading the MedYT corpus

yt-dlp --batch-file urls.txt -x --audio-format flac -o "%(channel)s/%(id)s/%(id)s.%(ext)s"


download human subtitles

python download_manual_subtitles.py \
    --urls urls.txt \
    --out-dir subtitles/ \
    --lang en \
    --workers 8 \
    --log subtitle_download.log
