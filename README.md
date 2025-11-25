## How to use

1. Run the search script to find videos with a filter
```bash
./scripts/search.py StandUp Comedy --output-format json --max-results 50 --min-length 600 --save-file ./dataset/videos.json 
```

2. Run the download script to download videos from the search results
```bash
./scripts/download.py ./dataset/videos.json --output-path ./dataset/video_downloads --shuffle --download-count 50 --only-audio
```

3.Or a one-liner
```bash
./scripts/download.py \
 <(./scripts/search.py \
   StandUp Comedy \
   --output-format json \
   --max-results 50 \
   --min-length 600) \
 --output-path ./dataset/video_downloads \
 --shuffle \
 --download-count 50 \
 --only-audio
```

4. Run the transcode script to extract audio from downloaded videos
```bash
./scripts/transcode.sh ./dataset/video_downloads ./dataset/audio_outputs webm wav
```
