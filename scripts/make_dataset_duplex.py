import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

try:
    from sca_data.dataset_utils import parse_aligned_script, duplex_data
except ImportError:
    print("[Error] we can't find the 'sca_data' package. Please ensure the 'src' directory is added to PYTHONPATH.")
    sys.exit(1)

DEFAULT_INPUT_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")
DEFAULT_OUTPUT_DIR = Path("./duplex_dataset")

def save_parsed_scripts(data_dir: Path, output_dir: Path):
    txt_dir = data_dir / "TXT"
    
    if not txt_dir.exists():
        raise FileNotFoundError(f"TXT folder not found in {data_dir}. Please check your input path.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> [Mode: Script] Parsing scripts from {txt_dir}")
    print(f">>> Saving JSON files to {output_dir}...")
    
    files = list(txt_dir.glob("*.txt"))
    if not files:
        print("[Warning] No .txt files found.")
        return

    for txt_file in tqdm(files, desc="Parsing & Saving"):
        events = parse_aligned_script(txt_file)
        
        save_name = txt_file.stem + ".json"
        save_path = output_dir / save_name
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
            
    print(f">>> Done! Parsed scripts saved at: {output_dir}")

def build_hf_dataset(data_dir: Path, output_dir: Path):
    print(f">>> [Mode: Dataset] Building Hugging Face Dataset from {data_dir}")
    print(f">>> Output Cache Dir: {output_dir}")
    
    # WAV, TXT 폴더 확인
    if not (data_dir / "WAV").exists() or not (data_dir / "TXT").exists():
         raise FileNotFoundError(f"Input directory must contain 'WAV' and 'TXT' folders.\nChecked: {data_dir}")
    ds = duplex_data(data_dir=data_dir, cache_dir=output_dir)
    
    print(f">>> Done! Full dataset saved at: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="SCA Dataset Converter Tool")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["script", "dataset"], 
        required=True,
        help="'script': 파싱된 JSON 스크립트 파일 저장 / 'dataset': 전체 데이터셋(Arrow) 빌드"
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=str(DEFAULT_INPUT_DIR),
        help=f"Raw Data Root Dir (contains WAV/TXT). Default: {DEFAULT_INPUT_DIR}"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(DEFAULT_OUTPUT_DIR), 
        help=f"Output directory path. Default: {DEFAULT_OUTPUT_DIR}"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"[Error] Input directory not found: {input_path}")
        print("Please ensure the 'Multi-stream Spontaneous Conversation Training Dataset' folder exists in the root.")
        sys.exit(1)

    if args.mode == "script":
        # 1. 스크립트 JSON 추출 모드
        save_parsed_scripts(input_path, output_path)
        
    elif args.mode == "dataset":
        # 2. 전체 데이터셋 빌드 모드
        build_hf_dataset(input_path, output_path)

if __name__ == "__main__":
    main()