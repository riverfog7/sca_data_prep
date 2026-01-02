#!/usr/bin/env -S uv run

import argparse
import sys
import traceback
from pathlib import Path

from sca_data.dataset_utils import remove_extras, assert_overlap, merge_close_events, to_hf_dataset
from sca_data.models.events import ComedySession

# --- Default Configuration Constants ---
DEFAULT_DATASET_DIR = Path("./dataset")
DEFAULT_INPUT_FILE = DEFAULT_DATASET_DIR / "inference_outputs.jsonl"
DEFAULT_AUDIO_DIR = DEFAULT_DATASET_DIR / "audio_outputs"
DEFAULT_SAVE_PATH = DEFAULT_DATASET_DIR / "sca_comedy_dataset"
DEFAULT_MERGE_THRESHOLD = 0.5
DEFAULT_MIN_DURATION = 1.0
DEFAULT_MAX_DURATION = float(3*60*60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and convert SCA Comedy inference outputs to a Hugging Face Dataset."
    )

    parser.add_argument(
        "-i", "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input JSONL file (default: {DEFAULT_INPUT_FILE})"
    )

    parser.add_argument(
        "-a", "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Directory containing source audio files (default: {DEFAULT_AUDIO_DIR})"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help=f"Path where the resulting Hugging Face dataset will be saved (default: {DEFAULT_SAVE_PATH})"
    )

    parser.add_argument(
        "-t", "--merge-threshold",
        type=float,
        default=DEFAULT_MERGE_THRESHOLD,
        help=f"Time threshold (in seconds) to merge close comedian events (default: {DEFAULT_MERGE_THRESHOLD})"
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help=f"Minimum duration (in seconds) for the audio"
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help=f"Maximum duration (in seconds) for the audio"
    )

    parser.add_argument(
        "--cut-start",
        type=int,
        default=3,
        help="Number of events to cut from the start of each audio file"
    )

    parser.add_argument(
        "--cut-end",
        type=int,
        default=1,
        help="Number of events to cut from the end of each audio file"
    )

    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=5.0,
        help="Minimum duration (in seconds) of speech segments within the audio"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_file = args.input_file.resolve()
    audio_dir = args.audio_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_file.exists():
        print(f"Error: Input file does not exist: {input_file}", file=sys.stderr)
        sys.exit(1)

    if not audio_dir.exists():
        print(f"Error: Audio directory does not exist: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing data from: {input_file}")
    print(f"Using audio source: {audio_dir}")
    print(f"Merge threshold: {args.merge_threshold}s")

    processed_sessions = []

    try:
        with open(input_file, "r") as f:
            for line_idx, line in enumerate(f):
                try:
                    validated = ComedySession.model_validate_json(line)
                    extras_removed = remove_extras(validated)
                    merged = merge_close_events(extras_removed, gap_threshold=args.merge_threshold)
                    assert_overlap(merged)
                    processed_sessions.append(merged)

                except Exception as e:
                    print(f"Warning: Failed to process line {line_idx + 1}: {e}", file=sys.stderr)
                    raise e

    except Exception as e:
        print(f"Critical Error during processing: {e}", file=sys.stderr)
        sys.exit(1)

    if not processed_sessions:
        print("Error: No sessions were processed. Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully processed {len(processed_sessions)} sessions.")

    print("Converting to Hugging Face Dataset format...")
    try:
        hf_dataset = to_hf_dataset(processed_sessions, audio_base_path=audio_dir,
                                   min_duration=args.min_duration, max_duration=args.max_duration,
                                   cut_start=args.cut_start, cut_end=args.cut_end,
                                   min_speech_duration=args.min_speech_duration)

        print(f"Saving dataset to: {output_dir}")
        hf_dataset.save_to_disk(output_dir)
        print("Done!")

    except Exception as e:
        print(f"Error creating/saving dataset: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()