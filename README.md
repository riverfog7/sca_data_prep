## How to use

1. Run the search script to find videos with a filter
```bash
uv run scripts/search.py StandUp Comedy --output-format json --max-results 50 --min-length 600 --save-file videos.json 
```

2. Run the download script to download videos from the search results
```bash
uv run scripts/download.py videos.json --output-path ./video_downloads --shuffle --download-count 50
```

2.5. Or a one-liner
```bash
uv run scripts/download.py \
 <(uv run scripts/search.py \
   StandUp Comedy \
   --output-format json \
   --max-results 50 \
   --min-length 600) \
 --output-path ./video_downloads \
 --shuffle \
 --download-count 50
```

3. Run the transcode script to extract audio from downloaded videos
```bash
./scripts/transcode.sh ./video_downloads ./audio_outputs
```
