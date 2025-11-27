## How to use
First run `uv sync --extra full,cu128` (cuda version depends on your GPU) to install the required dependencies.

1. Run the search script to find videos with a filter
```bash
./scripts/search.py StandUp Comedy English --output-format json --max-results 50 --min-length 600 --max-length 7200 --save-file ./dataset/videos.json 
```

2. Run the download script to download videos from the search results
```bash
./scripts/download.py ./dataset/videos.json --output-path ./dataset/video_downloads --shuffle --download-count 50 --only-audio
```

3.Or a one-liner
```bash
./scripts/download.py \
 <(./scripts/search.py \
   StandUp Comedy English \
   --output-format json \
   --max-results 50 \
   --min-length 600 \
   --max-length 7200) \
 --output-path ./dataset/video_downloads \
 --shuffle \
 --download-count 50 \
 --only-audio
```

4. Run the transcode script to extract audio from downloaded videos
```bash
./scripts/transcode.sh ./dataset/video_downloads ./dataset/audio_outputs webm flac
```

5. Run the inference code to generate transcripts from audio files
```bash
./main.py dataset/audio_outputs --save-file ./dataset/inference_outputs.jsonl
```

## Result files
Inference results are available in the following location:

[Full Dataset Files](https://web.aws.riverfog7.com/files/sca/dataset.tar.xz)

[Inference Output Files Only](https://web.aws.riverfog7.com/files/sca/inference_outputs.jsonl)
