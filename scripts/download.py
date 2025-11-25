import argparse
import random
import warnings
from pathlib import Path

import yt_dlp
from tqdm import tqdm

from data_models import SearchResult


def download_video(url: str, output_path: str = ".", only_audio=False) -> None:
    ydl_opts = {
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'format': 'bestaudio/best' if only_audio else 'bestvideo+bestaudio/best',
        'merge_output_format': 'webm' if only_audio else 'mkv',
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_videos_from_search(search_result: SearchResult, output_path: str = ".", only_audio=False) -> None:
    for entry in tqdm(search_result.results, desc="Downloading videos"):
        tqdm.write(f"Downloading: {entry.title} ({entry.url})")
        download_video(entry.url, output_path, only_audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos from search results.")
    parser.add_argument("search_results", type=str, help="Search results JSON file path.")
    parser.add_argument("--download-count", type=int, default=None, help="Number of videos to download from the search results.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the order of videos before downloading.")
    parser.add_argument("--output-path", type=str, default=".", help="Directory to save downloaded videos.")
    parser.add_argument("--only-audio", action="store_true", help="Download only audio from the videos.")
    args = parser.parse_args()

    search_results_path = Path(args.search_results)
    if not search_results_path.exists():
        raise FileNotFoundError(f"Search results file not found: {search_results_path}")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if not output_path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_path}")

    search_result = SearchResult.model_validate_json(search_results_path.read_text(encoding="utf-8"))
    if args.shuffle:
        random.shuffle(search_result.results)
    if args.download_count is not None:
        if int(args.download_count) > len(search_result.results):
            warnings.warn(f"Requested download count {args.download_count} exceeds available results {len(search_result.results)}. Downloading all available results.")
        search_result.results = search_result.results[:args.download_count]
    download_videos_from_search(search_result, output_path=str(output_path), only_audio=args.only_audio)
