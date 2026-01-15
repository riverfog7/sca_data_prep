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
import soundfile as sf  # 오디오 저장용 (없으면 pip install soundfile)
from transformers import Qwen3OmniMoeProcessor # 토크나이저 로드용
DEFAULT_INPUT_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")
DEBUG_OUTPUT_DIR = Path("./test_output") # 복원된 오디오/텍스트 저장 경로

NUM_SAMPLES_TO_CHECK = 100

# [Import]
try:
    from src.sca_data.dataset_utils import easy_load, DuplexConfig, AudioSeg, Audio
except ImportError:
    from sca_data.dataset_utils import easy_load, DuplexConfig, AudioSeg, Audio

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def verify_dataset():
    print_separator("데이터셋 로드 및 검증 시작")
    
    # -------------------------------------------------------------------------
    # 0. 토크나이저 로드 (디코딩용)
    # -------------------------------------------------------------------------
    print(">>> 토크나이저 로드 중 (Qwen/Qwen3-Omni-30B-A3B-Instruct)...")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", 
            trust_remote_code=True
        )
        tokenizer = processor.tokenizer
    except Exception as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}\n(텍스트 디코딩 기능이 작동하지 않을 수 있습니다)")
        tokenizer = None

    # -------------------------------------------------------------------------
    # 1. 데이터셋 로드
    # -------------------------------------------------------------------------
    try:
        ds = easy_load(DEFAULT_INPUT_DIR, format="duplex")
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
    SILENCE_TOKEN = 151646 # [중요] 생성 코드와 일치시킴
    AUDIO_RATIO = 4
    TEXT_SLICE = 2
    
    # 디버그 폴더 생성
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. 전체 데이터 순회
    # -------------------------------------------------------------------------
    for i, sample in enumerate(tqdm(ds, desc="검증 진행 중")):
        
        row = sample["dataset_row_obj"]
        seq_len = len(row.input_sequence)
        
        # [통계 집계]
        stats["max_seq_len"] = max(stats["max_seq_len"], seq_len)
        stats["min_seq_len"] = min(stats["min_seq_len"], seq_len)
        stats["total_tokens"] += seq_len
        
        # =====================================================================
        # [New Feature] 1. 시스템 프롬프트 & 텍스트 복원 검증 (Sample 0만 상세 출력)
        # =====================================================================
        if i == 0 and tokenizer is not None:
            print_separator(f"[Sample 0] 상세 디코딩 분석")
            
            # (1) 시스템 프롬프트 확인
            try:
                first_audio_idx = row.input_sequence.index(AUDIO_TOKEN)
                sys_prompt_ids = row.input_sequence[:first_audio_idx]
                sys_text = tokenizer.decode(sys_prompt_ids)
                print(f"🔹 [System Prompt 디코딩 결과]:\n{textwrap.fill(sys_text, width=80)}\n")
            except ValueError:
                print("❌ 오디오 토큰(-100)을 찾을 수 없어 시스템 프롬프트를 분리하지 못했습니다.")

            # (2) Target Audio <-> Text 매핑 복원
            print(f"🔹 [Target Audio & Text 매핑 복원] (총 {len(row.target_audios)}개 세그먼트 중 앞 5개만 출력)")
            
            full_reconstructed_text = []
            
            for seg_idx, seg in enumerate(row.target_audios):
                # 인덱스로 실제 토큰 ID 가져오기
                token_ids = [row.input_sequence[idx] for idx in seg.text_token_idxs]
                
                # 디코딩
                decoded_text = tokenizer.decode(token_ids)
                full_reconstructed_text.append(decoded_text)
                
                # 오디오 정보
                duration = len(seg.audio.waveform) / seg.audio.sampling_rate
                
                # 상위 5개만 로그 출력
                if seg_idx < 5:
                    print(f"  Start[{seg.text_token_idxs[0]}] -> Text: \"{decoded_text.strip()}\" | Audio: {duration:.2f}s")
                    
                    # (선택) 오디오 파일 저장 - 실제로 들어보고 싶으면 주석 해제
                    # sf.write(DEBUG_OUTPUT_DIR / f"sample0_seg{seg_idx}_{decoded_text.strip()[:10]}.wav", seg.audio.waveform, 24000)

            print(f"\n🔹 [전체 대화 흐름 복원]:\n{' '.join(full_reconstructed_text)[:200]} ... (중략)")
            print_separator("구조 검증 계속 진행")

        # =====================================================================
        # [Existing Checks] 기존 검증 로직 유지
        # =====================================================================
        
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
                continue

            if len(row.input_sequence[:first_audio_idx]) == 0:
                if stats["structure_error_count"] == 0:
                    print(f"\n❌ [Sample {i}] 시스템 프롬프트 누락")
                stats["structure_error_count"] += 1
            
            body_seq = row.input_sequence[first_audio_idx:]
            cursor = 0
            
            while cursor < len(body_seq):
                # (A) 오디오 4개
                audio_part = body_seq[cursor : cursor + AUDIO_RATIO]
                if len(audio_part) < AUDIO_RATIO: break 

                if not all(t == AUDIO_TOKEN for t in audio_part):
                    if stats["structure_error_count"] == 0:
                        print(f"\n❌ [Sample {i}] 오디오 패턴 깨짐: {audio_part}")
                    stats["structure_error_count"] += 1
                    break
                
                cursor += AUDIO_RATIO 

                # (B) 텍스트/침묵
                if cursor >= len(body_seq): break
                first_token = body_seq[cursor]

                if first_token == SILENCE_TOKEN:
                    cursor += 1 # 침묵 1개
                else:
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
    
    emb_status = "✅ 정상"
    if stats['zero_embedding_count'] > 0:
        emb_status = f"❌ 실패 ({stats['zero_embedding_count']} / {len(ds)} 샘플이 0으로 채워짐)"
    print(f"5. Speaker Embedding    : {emb_status}")

    if (stats['over_40k_count'] == 0 and 
        stats['sr_mismatch_count'] == 0 and 
        stats['structure_error_count'] == 0):
        print("\n🎉 [SUCCESS] 데이터셋 구조 검증 통과!")
    else:
        print("\n🔥 [FAILURE] 데이터셋에 문제가 있습니다. 요약을 확인하세요.")

if __name__ == "__main__":
    verify_dataset()