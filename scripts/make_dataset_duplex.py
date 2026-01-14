#!/usr/bin/env -S uv run python

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# [중요] transformers 추가 (토크나이저 로드용)
from transformers import Qwen3OmniMoeProcessor

# [Path 설정] src 폴더를 라이브러리 경로에 추가하여 sca_data 패키지 인식
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.append(str(src_path))

try:
    # 제공해주신 원본 코드(dataset_utils.py)에서 함수들을 가져옵니다.
    from sca_data.dataset_utils import parse_aligned_script, duplex_data
except ImportError as e:
    print(f"[Error] Failed to import 'sca_data'.")
    print(f"Debug info - sys.path: {sys.path}")
    print(f"Error details: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# [User Config] 기본 경로 설정 (여기서 고정합니다)
# -----------------------------------------------------------------------------
DEFAULT_INPUT_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")
DEFAULT_OUTPUT_DIR = Path("./Test_json")  # 요청하신 기본 출력 폴더명
DEFAULT_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# -----------------------------------------------------------------------------


def save_parsed_scripts(data_dir: Path, output_dir: Path, model_path: str):
    """
    [스크립트 모드]
    JSON 추출 및 토크나이징 확인용
    """
    txt_dir = data_dir / "TXT"

    if not txt_dir.exists():
        raise FileNotFoundError(f"TXT folder not found in {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f">>> [Mode: Script] Loading Tokenizer from {model_path}...")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
    except Exception as e:
        print(f"[Error] Failed to load model/tokenizer: {e}")
        sys.exit(1)

    print(f">>> Parsing scripts from {txt_dir} and saving to {output_dir}...")

    files = list(txt_dir.glob("*.txt"))
    if not files:
        print("[Warning] No .txt files found.")
        return

    for txt_file in tqdm(files, desc="Parsing & Saving"):
        # 수정된 parse_aligned_script는 tokenizer를 인자로 받음
        events = parse_aligned_script(txt_file, tokenizer)

        save_name = txt_file.stem + ".json"
        save_path = output_dir / save_name

        with open(save_path, "w", encoding="utf-8") as f:
            # input_ids가 포함된 결과 저장
            json.dump(events, f, ensure_ascii=False, indent=2)

    print(f">>> Done! JSON files saved at: {output_dir}")


def build_hf_dataset(data_dir: Path,  model_path: str,output_dir: Path = Path("./dataset")):
    """
    [데이터셋 모드]
    전체 데이터셋 빌드 (4만 토큰 경고 확인용)
    """
    print(f">>> [Mode: Dataset] Building Hugging Face Dataset from {data_dir}")
    print(f">>> Output Cache Dir: {output_dir}")
    print(f">>> Model Path: {model_path}")

    if not (data_dir / "WAV").exists() or not (data_dir / "TXT").exists():
        raise FileNotFoundError(f"Input directory must contain 'WAV' and 'TXT' folders inside {data_dir}")

    # duplex_data 함수 호출
    duplex_data(data_dir=data_dir, cache_dir=output_dir, model_path=model_path)

    print(f">>> Done! Full dataset saved at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SCA Duplex Dataset Converter")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["script", "dataset"],
        required=True,
        help="'script': JSON 추출(토크나이징 확인) / 'dataset': 전체 데이터셋 빌드(경고 확인)",
    )

    # 기본값을 변수로 설정하여 인자를 안 줘도 자동으로 적용됨
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"입력 데이터 경로 (Default: {DEFAULT_INPUT_DIR})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"결과 저장 경로 (Default: {DEFAULT_OUTPUT_DIR})",
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Qwen 모델 경로",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[Error] Input directory not found: {input_path}")
        print(f"현재 위치: {Path.cwd()}")
        sys.exit(1)

    if args.mode == "script":
        save_parsed_scripts(input_path, output_path, args.model_path)

    elif args.mode == "dataset":
        build_hf_dataset(input_path, args.model_path)


if __name__ == "__main__":
    main()