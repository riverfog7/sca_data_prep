import sys
import os

# src 폴더를 파이썬 경로에 추가 (sca_data 패키지를 찾기 위함)
sys.path.append(os.path.join(os.getcwd(), "src"))

import numpy as np
from pathlib import Path

# ★ 수정된 부분: 패키지 전체 경로로 임포트
from sca_data.dataset_utils import duplex_data
# 1. 데이터셋 경로 설정 (사용자 환경에 맞게 수정 필요)
# 예: 현재 폴더에 'dataset' 폴더가 있고 그 안에 'WAV', 'TXT' 폴더가 있다고 가정
DATA_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")

def print_sample_details(idx, sample):
    print(f"\n{'='*20} Sample [{idx}] {'='*20}")
    print(f"Session ID: {sample['session_id']}")
    
    # 기본 길이 정보
    num_chunks = len(sample['types'])
    print(f"Total Chunks: {num_chunks}")
    
    # 데이터 내용 미리보기 (앞부분 10개만 출력)
    print("\n--- [Preview: First 10 Items] ---")
    
    types = sample['types'][:10]
    texts = sample['texts'][:10]
    masks = sample['label_mask'][:10]
    waveforms = sample['waveforms'][:10]
    
    for i in range(len(types)):
        # Waveform 정보 (Shape 및 0인지 아닌지)
        wav_arr = np.array(waveforms[i])
        wav_info = f"Shape={wav_arr.shape}, Max={np.max(np.abs(wav_arr)):.4f}"
        if np.all(wav_arr == 0):
            wav_info += " (Silence/Dummy)"
            
        # 텍스트 정보 (있으면 출력)
        txt_info = f'"{texts[i]}"' if texts[i] else "-"
        
        # 출력 포맷
        # Type | Mask | Text | Audio Info
        print(f"[{i}] Type: {types[i]:<12} | Mask: {masks[i]:<4} | Text: {txt_info:<15} | Audio: {wav_info}")

    # 통계 정보
    total_user = sample['types'].count('user_audio')
    total_target = sample['types'].count('target_audio')
    total_text = sample['types'].count('text')
    
    print(f"\n--- [Statistics] ---")
    print(f"User Audio Chunks: {total_user}")
    print(f"Target Audio Chunks: {total_target}")
    print(f"Text Chunks (Think): {total_text}")
    print(f"Label Mask (-100): {sample['label_mask'].count(-100)}")
    print(f"Label Mask (1):    {sample['label_mask'].count(1)}")

def main():
    # 데이터 폴더 확인
    if not DATA_DIR.exists():
        print(f"Error: 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        print("test.py 파일 안의 'DATA_DIR' 변수를 실제 데이터 경로로 수정해주세요.")
        return

    print(">>> 데이터셋 생성 중... (WAV/TXT 로딩)")
    # sample_rate는 코드상의 기본값(16000) 사용
    dataset = duplex_data(DATA_DIR)
    
    print(f">>> 데이터셋 로드 완료! 총 시퀀스 개수: {len(dataset)}")
    
    # 0~5번 샘플 출력
    # 데이터셋이 적을 경우를 대비해 min 처리
    max_idx = min(6, len(dataset))
    
    for i in range(max_idx):
        sample = dataset[i]
        #print(sample)
        print_sample_details(i, sample)

if __name__ == "__main__":
    main()