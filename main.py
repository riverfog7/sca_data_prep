#!/usr/bin/env -S uv run

import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

from sca_data.constants import PROCESS_EXTS
from sca_data.inference import infer_comedy_session


def process_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    session = infer_comedy_session(file_path)
    return session.model_dump_json(ensure_ascii=False)


def process_dir(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    results = []
    for file_path in tqdm(list(dir_path.glob("*")), desc="Processing files"):
        if file_path.is_file():
            if file_path.suffix.lower() not in PROCESS_EXTS:
                tqdm.write(f"Skipping unsupported file type: {file_path}")
                continue
            tqdm.write(f"Processing file: {file_path}")
            results.append(process_file(file_path))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer comedy sessions from audio files.")
    parser.add_argument("input_path", type=str, help="Path to an audio file or a directory containing audio files.")
    parser.add_argument("--save-file", type=str, default=None, help="Optional path to save the output JSON (or JSONL).")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if input_path.is_file():
        output_json = process_file(input_path)
        if args.save_file:
            with open(args.save_file, "w", encoding="utf-8") as f:
                f.write(output_json)
        else:
            print(output_json)
    elif input_path.is_dir():
        output_jsons = process_dir(input_path)
        if args.save_file:
            with open(args.save_file, "w", encoding="utf-8") as f:
                for json_str in output_jsons:
                    f.write(json_str + "\n")
        else:
            for json_str in output_jsons:
                print(json_str)
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")
