import sys
import os
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------------
# 1. 경로 설정 (src 폴더를 파이썬 라이브러리 경로에 추가)
# -------------------------------------------------------------------------
current_dir = os.getcwd()
src_path = os.path.join(current_dir, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

try:
    # easy_load를 통해 데이터셋을 로드합니다.
    from sca_data.dataset_utils import easy_load
except ImportError as e:
    print("\n[Critical Error] sca_data 패키지를 찾을 수 없습니다.")
    raise e

# -------------------------------------------------------------------------
# 2. 데이터 경로 설정
# -------------------------------------------------------------------------
# 실제 데이터가 위치한 폴더명으로 수정해주세요.
DATA_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")

def print_sample_details(idx, sample):
    """
    새로운 Pydantic 구조(DatasetRow)를 분석하여 출력하는 함수
    """
    print(f"\n{'='*20} Step [{idx}] {'='*20}")
    
    # 1. DatasetRow 객체 가져오기
    # transform에서 반환한 dict의 키는 "dataset_row_obj"입니다.
    row = sample['dataset_row_obj']
    
    # 2. Input History (User + Text 누적) 확인
    input_blocks = row.input
    num_blocks = len(input_blocks)
    
    print(f"1. Input History Length: {num_blocks} blocks")
    print(f"   (Expectation: Step이 지날수록 길이가 2씩(User+Text) 늘어나야 함)")

    # 3. Interleaved Structure (교차 구조) 확인 - 마지막 4개 블록만 미리보기
    print(f"\n2. Input Blocks Preview (Last 4 items):")
    print(f"   {'Index':<5} | {'Type':<15} | {'Content Info'}")
    print("   " + "-" * 60)
    
    start_view = max(0, num_blocks - 4)
    for i in range(start_view, num_blocks):
        block = input_blocks[i]
        b_type = block.type
        
        content_info = ""
        if b_type == "user_audio":
            # Audio 객체 확인
            wav_shape = block.audio.waveform.shape
            sr = block.audio.sampling_rate
            content_info = f"Wave: {wav_shape}, SR: {sr}Hz"
        elif b_type == "target_text":
            # Text 내용 확인 (빈 문자열일 수도 있음)
            txt = block.text
            content_info = f'Text: "{txt}"' if txt else '(Empty String)'
            
        print(f"   {i:<5} | {b_type:<15} | {content_info}")

    # 4. Target Audio (정답) 확인
    target_wav = row.target_audio
    print(f"\n3. Target Audio Info (Label):")
    print(f"   Shape: {target_wav.shape}")
    
    # 검증: 24kHz * 0.32s = 7680 샘플이어야 함
    expected_samples = int(24000 * 0.32)
    if target_wav.shape[0] == expected_samples:
        print(f"   >>> [PASS] Target Shape matches 24kHz 4-token spec ({expected_samples}).")
    else:
        print(f"   >>> [WARNING] Target Shape mismatch! Expected {expected_samples}, got {target_wav.shape[0]}")


def main():
    print(f">>> 데이터셋 로드 시작 (Format: Duplex, Path: {DATA_DIR})...")
    
    # 1. 데이터셋 로드 (최초 실행 시 Preprocess가 동작할 수 있음)
    # format="duplex"로 설정하여 DuplexTransform이 적용된 데이터셋을 받습니다.
    try:
        dataset = easy_load(dataset_path=DATA_DIR,format="duplex")
    except Exception as e:
        print(f">>> [오류] 데이터셋 로드 중 에러 발생: {e}")
        return

    # 2. 전체 길이 확인
    total_len = len(dataset)
    print(f">>> [성공] 데이터셋 로드 완료! 총 데이터(Steps) 개수: {total_len}")
    
    # 3. 순차적 접근 테스트 (Accumulation 확인)
    print("\n>>> 샘플 데이터 조회 테스트 (연속된 3개의 스텝 확인)...")
    
    # 0, 1, 2번 인덱스를 순서대로 찍어봅니다.
    # 인덱스가 증가할수록 Input History Length가 늘어나는지 확인하세요.
    test_indices = [0, 1, 2, 10] 
    
    for i in test_indices:
        if i >= total_len:
            break
        
        # 여기서 __call__ (Lazy Loading & Processing)이 실행됩니다.
        sample = dataset[i] 
        print_sample_details(i, sample)
        
    print("\n>>> 테스트 종료.")

if __name__ == "__main__":
    main()