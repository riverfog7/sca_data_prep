## How to use
First run `uv sync --extra full,cu128` (cuda version depends on your GPU, for CPU only, use the `cpu` extras) to install the required dependencies.

1. Run the search script to find videos with a filter
```bash
./scripts/search.py StandUp Comedy English --output-format json --max-results 50 --min-length 900 --max-length 2400 --save-file ./dataset/videos.json 
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
   --min-length 900 \
   --max-length 2400) \
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

6. (Optional) Run the convert_dataset script to convert the dataset to a huggingface format
```bash
./scripts/convert_dataset.py --merge-threshold 0.5
```

## full-duplex finetune dataset
We constructed this dataset to modify the existing half-duplex model into a full-duplex architecture, aiming to enhance the real-time spontaneity of the comedian.
**Technical Specifications:**
* **Architecture:** Full-Duplex (Simultaneous listening and speaking)
* **Chunking:** Split into **4-token chunks** based on a 12.5Hz encoder (approx. 0.32s).
* **Text Alignment:** Text is aligned to 2 tokens.
* **Audio Sampling Rate:**
  * **Input ($x$):** 16kHz (User/Environment history)
  * **Target ($y$):** 24kHz (Comedian response)


> **Note:** If you want to verify the implementation details yourself instead of using the pre-processed dataset, you must download the raw data and place it in the root directory.

### 1. Prerequisite: Raw Data Setup
1. Download the **Multi-stream Spontaneous Conversation Training Dataset** from [MagicHub](https://magichub.com/datasets/multi-stream-spontaneous-conversation-training-datasets_english/).
2. Place the dataset folder in the root directory.
3. Ensure the folder contains both `WAV` and `TXT` subdirectories.

#### 2. Usage
**Option A: Generate Script Files Only**
If you want to extract and verify the script parsing logic (saves as JSON):
```bash
python scripts/convert_dataset.py --mode script --output ./parsed_json
```

**Option B: Build HuggingFace Dataset**
To convert the raw data into the full HuggingFace dataset structure (Arrow format):
```bash
python scripts/convert_dataset.py --mode dataset
```
## Result files
Inference results are available in the following location:

[Inference Output Files Only](https://web.aws.riverfog7.com/files/sca/inference_outputs.jsonl)

[Huggingface Datasets format dataset](https://web.aws.riverfog7.com/files/sca/sca_comedy_dataset.tar)

[Huggingface Datasets format dataset_for_<full-duplex>](https://huggingface.co/datasets/wjm9765/sca_full_duplex/resolve/main/sca_duplex_cache.tar?download=true)


