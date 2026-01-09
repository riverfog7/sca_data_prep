import sys
import os
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------------
# 1. 경로 설정 (src 폴더를 파이썬 라이브러리 경로에 추가)
# -------------------------------------------------------------------------
# 현재 실행 위치(getcwd) 기준으로 src 폴더를 찾습니다.
current_dir = os.getcwd()
src_path = os.path.join(current_dir, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

# 경로 추가 후 import
try:
    from sca_data.dataset_utils import duplex_data
except ImportError as e:
    print("\n[Critical Error] sca_data 패키지를 찾을 수 없습니다.")
    print(f"현재 경로: {current_dir}")
    print(f"예상되는 src 경로: {src_path}")
    print("dataset_utils.py 파일이 ./src/sca_data/ 폴더 안에 있는지 확인해주세요.\n")
    raise e

# -------------------------------------------------------------------------
# 2. 데이터 경로 설정
# -------------------------------------------------------------------------
# 실제 데이터가 위치한 폴더명으로 수정해주세요.
DATA_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")

def print_sample_details(idx, sample):
    """
    단일 샘플(딕셔너리)을 받아서 내부 구조를 보기 좋게 출력하는 함수
    """
    print(f"\n{'='*20} Sample [{idx}] {'='*20}")
    
    # Transform을 거치면 session_id 컬럼이 사라질 수 있으므로 .get() 사용
    sess_id = sample.get('session_id', 'Unknown (Transformed)')
    print(f"Session ID: {sess_id}")
    
    # 1. 전체 청크 개수 확인
    # sample['types']는 리스트 형태여야 합니다.
    types = sample['types']
    num_chunks = len(types)
    print(f"Total Chunks: {num_chunks}")
    
    # 2. 통계 정보 출력
    print(f"\n--- [Statistics] ---")
    print(f"User Audio:   {types.count('user_audio')}")
    print(f"Target Audio: {types.count('target_audio')}")
    print(f"Text (Think): {types.count('text')}")
    
    # 3. 앞부분 데이터 미리보기 (최대 10개)
    print("\n--- [Preview: First 10 Steps] ---")
    
    texts = sample['texts']
    masks = sample['label_mask']
    waveforms = sample['waveforms']
    
    preview_len = min(10, num_chunks)
    
    print(f"{'Index':<5} | {'Type':<12} | {'Mask':<4} | {'Audio Shape':<20} | {'Text / Content'}")
    print("-" * 80)
    
    for i in range(preview_len):
        # 오디오 정보 확인
        wav_arr = np.array(waveforms[i])
        wav_shape_str = str(wav_arr.shape)
        
        # 텍스트 정보 확인
        txt_content = f'"{texts[i]}"' if texts[i] else ""
        
        # 마스크 정보
        mask_val = masks[i]
        
        print(f"{i:<5} | {types[i]:<12} | {mask_val:<4} | {wav_shape_str:<20} | {txt_content}")

def main():
    # 데이터 폴더 존재 여부 확인
    if not DATA_DIR.exists():
        print(f"\n[Error] 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        print("test.py 코드 상단의 'DATA_DIR' 변수를 실제 폴더 경로로 수정해주세요.\n")
        return

    print(f">>> 데이터셋 로드 시작 (경로: {DATA_DIR})...")
    
    # 1. 데이터셋 로드 함수 호출
    # duplex_data는 Dataset 객체(train split)를 반환해야 합니다.
    dataset = duplex_data(DATA_DIR)
    
    # 2. len() 테스트
    try:
        total_len = len(dataset)
        print(f">>> [성공] 데이터셋 로드 완료! 총 시퀀스 개수: {total_len}")
    except TypeError as e:
        print(f">>> [오류] len() 호출 실패. dataset 객체가 리스트나 제너레이터인지 확인하세요.")
        print(f"반환된 타입: {type(dataset)}")
        raise e
    
    # 3. 데이터 접근 테스트 (인덱싱)
    print("\n>>> 샘플 데이터 조회 테스트 (0번 ~ 2번)...")
    
    # 데이터가 적을 경우를 대비해 min 사용
    test_count = min(10, total_len)
    
    for i in range(test_count):
        # 여기서 Transform이 실행됩니다 (__call__)
        sample = dataset[i] 
        print_sample_details(i, sample)

if __name__ == "__main__":
    main()