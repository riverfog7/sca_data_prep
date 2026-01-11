import sys
import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pathlib import Path

# -------------------------------------------------------------------------
# 1. 경로 설정
# -------------------------------------------------------------------------
current_dir = os.getcwd()
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from sca_data.dataset_utils import easy_load

# -------------------------------------------------------------------------
# 2. 설정
# -------------------------------------------------------------------------
OUTPUT_DIR = Path("./test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 확인하고 싶은 인덱스 번호 (에러가 났던 80번이나, 데이터가 충분한 100번 등)
SAMPLE_IDX = 100 

TARGET_SR = 24000 
CHUNK_DURATION = 0.32

def main():
    print(">>> 데이터셋 로드 중...")
    # data_dir 없이 dataset_path나 format만 줘도 캐시된 데이터를 로드합니다.
    dataset = easy_load(format="duplex")
    
    total_len = len(dataset)
    print(f">>> 데이터셋 로드 완료. 총 길이: {total_len}")
    
    if SAMPLE_IDX >= total_len:
        print(f"[Error] 인덱스 {SAMPLE_IDX}는 범위를 벗어났습니다. (최대 {total_len-1})")
        return

    print(f">>> {SAMPLE_IDX}번 샘플 데이터를 디코딩합니다...")
    
    # 1. 데이터 가져오기 (Lazy Loading)
    # dataset[i]는 이제 {'dataset_row_obj': DatasetRow(...)} 형태입니다.
    row = dataset[SAMPLE_IDX]['dataset_row_obj']
    
    # Input History (User Audio + Text)
    input_blocks = row.input
    # Target Audio (Label)
    target_wav_24k = row.target_audio
    
    print(f"   - Input History Blocks: {len(input_blocks)}개")
    print(f"   - Target Audio Shape: {target_wav_24k.shape}")

    # 2. 오디오 스트림 구성
    # Left: User Audio (Input History)
    # Right: Target Audio (Label) -> 마지막에만 등장
    
    left_channel = []  # User
    text_log = []
    
    # 리샘플러 (User 16k -> 24k)
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)
    
    print(">>> Input History 복원 중...")
    
    # 시간 추적용
    current_time = 0.0
    
    for idx, block in enumerate(input_blocks):
        if block.type == "user_audio":
            # 16k Audio -> 24k Resample
            wav_16k = block.audio.waveform
            
            # 텐서 변환 및 리샘플링
            t_in = torch.from_numpy(wav_16k).unsqueeze(0) # [1, T]
            t_out = resampler(t_in).squeeze(0).numpy()    # [T]
            
            left_channel.extend(t_out)
            current_time += CHUNK_DURATION
            
        elif block.type == "target_text":
            if block.text:
                # 텍스트는 오디오 바로 직전 시간에 위치함
                log_line = f"[{current_time - CHUNK_DURATION:.2f}s ~ {current_time:.2f}s] Text: {block.text}"
                text_log.append(log_line)

    # 3. 채널 병합
    # Left Channel: User History 전체
    # Right Channel: 마지막 Target 부분에만 소리가 나고, 앞부분은 무음(Silence) 처리
    # (왜냐하면 Input Feature에는 과거의 Target Audio가 포함되어 있지 않기 때문입니다)
    
    len_user = len(left_channel)
    len_target = len(target_wav_24k)
    
    # Right 채널: 앞부분은 0으로 채우고, 끝에 Target Audio 붙이기
    right_channel = np.zeros(len_user, dtype=np.float32)
    
    # 최종 길이는 User History 길이 + 이번 Target 길이 (선택사항이나, 보통 겹쳐서 듣거나 이어서 들음)
    # 여기서는 "이어 듣기" 형태로 만들겠습니다. (User 다 듣고 -> Target 정답 듣기)
    
    # Left: [User History ............] [Silence (Target 구간)]
    final_left = np.concatenate([left_channel, np.zeros(len_target, dtype=np.float32)])
    
    # Right: [Silence (User 구간).......] [Target Audio]
    final_right = np.concatenate([np.zeros(len_user, dtype=np.float32), target_wav_24k])
    
    stereo_audio = np.stack([final_left, final_right], axis=1)
    
    # 4. 저장
    wav_name = f"sample_{SAMPLE_IDX}_input_vs_target.wav"
    txt_name = f"sample_{SAMPLE_IDX}_text.txt"
    
    wav_path = OUTPUT_DIR / wav_name
    txt_path = OUTPUT_DIR / txt_name
    
    sf.write(wav_path, stereo_audio, TARGET_SR)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Sample Index: {SAMPLE_IDX}\n")
        f.write("-" * 40 + "\n")
        f.write("\n".join(text_log))
        f.write(f"\n\n[End of Input History]\n")
        f.write(f">>> Next Target Audio (Right Channel) plays now.")

    print(f"\n>>> [완료]")
    print(f"1. 오디오 파일: {wav_path}")
    print(f"   ★ 듣는 법:")
    print(f"   - 왼쪽 소리(User)가 쭉 나오다가, 끝나면 오른쪽 소리(Target)가 '삑' 하고 나옵니다.")
    print(f"   - 이것이 모델이 학습하는 [Input -> Output] 관계입니다.")
    print(f"2. 텍스트 로그: {txt_path}")

if __name__ == "__main__":
    main()