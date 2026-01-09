import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path

# src 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))
from sca_data.dataset_utils import duplex_data,easy_load

# 설정
OUTPUT_DIR = Path("./test_output")
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.16
DURATION_LIMIT_SEC = 180  # 3분

def main():
    print(">>> 데이터셋 로드 중...")
    # 캐시가 이미 있다면 data_dir은 None이어도 됨
    dataset = easy_load(format="duplex")
    
    sample_idx = 80
    print(f">>> {sample_idx}번 샘플 복원 및 로그 저장 시작...")
    
    sample = dataset[sample_idx]
    
    types = sample['types']
    waveforms = sample['waveforms']
    texts = sample['texts']
    
    user_stream = []
    target_stream = []
    full_text_log = []  # 전체 로그를 담을 리스트
    
    # 청크 단위 순회
    limit_chunks = int(DURATION_LIMIT_SEC / CHUNK_DURATION) * 2
    
    current_time = 0.0
    
    # tqdm 없이 빠르게 처리
    loop_range = min(len(types), limit_chunks)
    
    for i in range(loop_range):
        t_type = types[i]
        waveform = np.array(waveforms[i])
        
        # 1. 텍스트 로그 수집 (모든 텍스트 기록)
        if t_type == "text":
            if texts[i]:
                log_line = f"[{current_time:.2f}s] {texts[i]}"
                full_text_log.append(log_line)
            continue
            
        # 2. 오디오 데이터 수집
        if t_type == "user_audio":
            user_stream.extend(waveform)
            current_time += CHUNK_DURATION
            
        elif t_type == "target_audio":
            target_stream.extend(waveform)
            
    # 길이 맞추기
    min_len = min(len(user_stream), len(target_stream))
    user_stream = np.array(user_stream[:min_len])
    target_stream = np.array(target_stream[:min_len])
    
    # Stereo Audio 저장
    stereo_audio = np.stack([user_stream, target_stream], axis=1)
    wav_path = OUTPUT_DIR / f"reconstructed_sample_{sample_idx}.wav"
    sf.write(wav_path, stereo_audio, SAMPLE_RATE)
    
    # ★ 핵심: 전체 텍스트 로그를 파일로 저장
    txt_path = OUTPUT_DIR / f"reconstructed_sample_{sample_idx}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in full_text_log:
            f.write(line + "\n")
    
    print(f"\n>>> [완료]")
    print(f"1. 오디오 파일: {wav_path} (길이: {min_len/SAMPLE_RATE:.2f}초)")
    print(f"2. 전체 텍스트 로그: {txt_path} (총 {len(full_text_log)} 라인)")
    print("   -> 이제 텍스트 파일을 열어보시면 끝까지 다 나와있을 겁니다.")

if __name__ == "__main__":
    main()