# 1. 하나의 데이터셋 dataset[i]가 4만 토큰 이내인지 체크
# 2. target audio에 의미 없는 1초 이하 오디오가 있는지 체크 (너무 짧은 오디오, 문장 등)
# 3. user 샘플링 16000Hz, assistant 샘플링 24000Hz 체크
# 4. 전체 구조가 의도한대로 나왔는지 체크 
# 5. speaker_embedding이 제대로 있는지, 실패해서 0으로 채워지지 않았는지 
# 6. 시스템프롬프트 있는지 , 시퀀스 구조가 맞는지 4 2 4 2 4 2 .. 
#7 . target_audio 는 어떻게 저장되어있는지 확인 

#!/usr/bin/env -S uv run python
#!/usr/bin/env -S uv run python

import numpy as np
from pathlib import Path
from tqdm import tqdm
import textwrap
DEFAULT_INPUT_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")

# [Import]
try:
    from src.sca_data.dataset_utils import easy_load, DuplexConfig, AudioSeg, Audio
except ImportError:
    from sca_data.dataset_utils import easy_load, DuplexConfig, AudioSeg, Audio
NUM_SAMPLES_TO_CHECK = 100  # <--- 여기를 수정하세요! (예: 100개만 확인)
def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def verify_dataset():
    print_separator("데이터셋 로드 및 검증 시작")
    
    try:
        ds = easy_load(DEFAULT_INPUT_DIR,format="duplex")

        total_len = len(ds)
        if NUM_SAMPLES_TO_CHECK is not None and NUM_SAMPLES_TO_CHECK < total_len:
            print(f"✂️  설정에 따라 앞부분 {NUM_SAMPLES_TO_CHECK}개만 잘라서 검증합니다.")
            ds = ds.select(range(NUM_SAMPLES_TO_CHECK))
        print(f"✅ 데이터셋 로드 성공! 총 샘플 수: {len(ds)}")
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return
# 통계 변수
    stats = {
        "max_seq_len": 0,
        "min_seq_len": 999999,
        "total_tokens": 0,
        "over_40k_count": 0,
        "short_target_audio_count": 0, 
        "zero_embedding_count": 0,
        "sr_mismatch_count": 0,
        "structure_error_count": 0,
    }

    # 상수 설정
    AUDIO_TOKEN = -100
    SILENCE_TOKEN = 151643
    AUDIO_RATIO = 4
    TEXT_SLICE = 2
    
    inspected_target_structure = False

    # 2. 전체 데이터 순회
    for i, sample in enumerate(tqdm(ds, desc="검증 진행 중")):
        
        row = sample["dataset_row_obj"]

        # ---------------------------------------------------------------------
        # [기능 추가] 토큰 수 카운트 및 통계
        # ---------------------------------------------------------------------
        seq_len = len(row.input_sequence)
        stats["max_seq_len"] = max(stats["max_seq_len"], seq_len)
        stats["min_seq_len"] = min(stats["min_seq_len"], seq_len)
        stats["total_tokens"] += seq_len
        
        # 1. 길이 4만 토큰 체크
        if seq_len > 40000:
            stats["over_40k_count"] += 1
            if stats["over_40k_count"] == 1:
                print(f"\n❌ [Sample {i}] 길이 초과 발견: {seq_len} tokens")

        # 2. Target Audio 1초 이하 체크
        for audio_seg in row.target_audios:
            duration = len(audio_seg.audio.waveform) / 24000.0
            if duration < 1.0:
                stats["short_target_audio_count"] += 1

        # 3. 샘플링 레이트 체크
        if row.input_audios and row.input_audios[0].sampling_rate != 16000:
            stats["sr_mismatch_count"] += 1
        
        if row.target_audios and row.target_audios[0].audio.sampling_rate != 24000:
            stats["sr_mismatch_count"] += 1

        # 5. Speaker Embedding 체크
        # (최초 1회만 경고 출력, 나머지는 카운트만 함)
        emb = np.array(row.speaker_embedding)
        if np.all(emb == 0):
            stats["zero_embedding_count"] += 1
            if stats["zero_embedding_count"] == 1:
                print(f"\n❌ [Sample {i}] Speaker Embedding이 모두 0입니다. (이후 생략)")

        # 6. 구조 패턴 체크 (Silence 1개, Text 2개 동적 대응)
        try:
            try:
                first_audio_idx = row.input_sequence.index(AUDIO_TOKEN)
            except ValueError:
                # 오디오가 없는 경우
                continue

            # 시스템 프롬프트 확인
            if len(row.input_sequence[:first_audio_idx]) == 0:
                if stats["structure_error_count"] == 0:
                    print(f"\n❌ [Sample {i}] 시스템 프롬프트 누락")
                stats["structure_error_count"] += 1
            
            # 본문 패턴 확인
            body_seq = row.input_sequence[first_audio_idx:]
            cursor = 0
            
            while cursor < len(body_seq):
                # (A) 오디오 4개 확인
                audio_part = body_seq[cursor : cursor + AUDIO_RATIO]
                if len(audio_part) < AUDIO_RATIO: break 

                if not all(t == AUDIO_TOKEN for t in audio_part):
                    if stats["structure_error_count"] == 0:
                        print(f"\n❌ [Sample {i}] 오디오 패턴 깨짐: {audio_part}")
                    stats["structure_error_count"] += 1
                    break
                
                cursor += AUDIO_RATIO 

                # (B) 텍스트/침묵 확인
                if cursor >= len(body_seq): break
                first_token = body_seq[cursor]

                if first_token == SILENCE_TOKEN:
                    cursor += 1 # 침묵은 1개
                else:
                    # 텍스트는 2개 (오디오 토큰 끼어있으면 에러)
                    text_part = body_seq[cursor : cursor + TEXT_SLICE]
                    if len(text_part) < TEXT_SLICE: break 
                    if any(t == AUDIO_TOKEN for t in text_part):
                        if stats["structure_error_count"] == 0:
                            print(f"\n❌ [Sample {i}] 텍스트 패턴 깨짐: {text_part}")
                        stats["structure_error_count"] += 1
                        break
                    cursor += TEXT_SLICE 

        except Exception as e:
            if stats["structure_error_count"] == 0:
                print(f"\n❌ [Sample {i}] 검증 중 예외 발생: {e}")
            stats["structure_error_count"] += 1

    # ---------------------------------------------------------------------
    # 최종 리포트 출력
    # ---------------------------------------------------------------------
    avg_len = stats["total_tokens"] / len(ds) if len(ds) > 0 else 0
    
    print_separator("📊 토큰 길이 통계")
    print(f"▶ 최소 길이: {stats['min_seq_len']} tokens")
    print(f"▶ 최대 길이: {stats['max_seq_len']} tokens (Limit: 40000)")
    print(f"▶ 평균 길이: {avg_len:.2f} tokens")

    print_separator("🛠 검증 결과 요약")
    print(f"1. 4만 토큰 초과 샘플 수 : {stats['over_40k_count']} 개")
    print(f"2. 구조 패턴 에러 샘플 수 : {stats['structure_error_count']} 개")
    print(f"3. SR 불일치 샘플 수    : {stats['sr_mismatch_count']} 개")
    print(f"4. 1초 미만 오디오 개수  : {stats['short_target_audio_count']} 개 (참고용)")
    
    # 임베딩 결과 확인
    emb_status = "✅ 정상"
    if stats['zero_embedding_count'] > 0:
        emb_status = f"❌ 실패 ({stats['zero_embedding_count']} / {len(ds)} 샘플이 0으로 채워짐)"
    print(f"5. Speaker Embedding    : {emb_status}")

    # 최종 판정
    if (stats['over_40k_count'] == 0 and 
        stats['sr_mismatch_count'] == 0 and 
        stats['structure_error_count'] == 0):
        print("\n🎉 [SUCCESS] 데이터셋 구조 검증 통과!")
    else:
        print("\n🔥 [FAILURE] 데이터셋에 문제가 있습니다. 요약을 확인하세요.")

if __name__ == "__main__":
    verify_dataset()