import io
import json
import math
import re
import shutil
import tarfile
from hashlib import md5
from pathlib import Path
from typing import Tuple, Optional, Iterable, Literal

import numpy as np
import requests
import soundfile as sf
import torchaudio
from datasets import DatasetDict, Dataset, load_from_disk
from datasets import Features, Value, Audio as HFAudio, Sequence
from pydantic import BaseModel
from tqdm import tqdm
from dataclasses import dataclass #dataclass annotation 추가

from transformers import Qwen3OmniMoeProcessor

from .constants import DEFAULT_SYSTEM_PROMPT, DEFAULT_INSTRUCTION_PROMPT
from .models.events import (
    ComedianEvent,
    BaseEvent,
    AudienceEvent,
    EnvironmentEvent,
    ComedySession,
)
from .utils import (
    clean_audio_bytes,
    check_and_resample_audio,
    extract_speaker_embedding,
    SPEAKER_EMBEDDING_DIM,
)

# 설정: 12.5 기준 4토큰
CHUNK_DURATION = 0.32
SAMPLE_RATE_USER = 16000
SAMPLE_RATE_TARGET = 24000

@dataclass
class DuplexConfig:
    audio_placeholder_token: int = -100 
    audio_token_ratio: int = 4          # [Modified] 오디오 청크 당 -100 개수 (4개)
    text_token_slice_len: int = 2       # 청크당 가져올 텍스트 토큰 수
    silence_token_id: int = 151646     # 묵음 토큰 151646~151651
    max_token_length: int = 40000       
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


@dataclass
class Audio:
    waveform: np.ndarray
    sampling_rate: int

@dataclass
class AudioSeg:
    text_token_idxs: list[int]  # 전체 input_sequence 내에서 해당 오디오에 대응하는 텍스트가 위치한 인덱스들
    audio: Audio

@dataclass
class DatasetRow:
    # LLM에 들어갈 최종 입력 시퀀스 (Audio Placeholder Token + Text Token이 인터리빙된 형태)
    input_sequence: list[int]    
    # Loss 계산 및 음성 합성을 위한 타겟 오디오 리스트
    target_audios: list[AudioSeg]
    input_audios: list[Audio]
    # 화자 임베딩 (192차원 등)
    speaker_embedding: np.ndarray


def preprocess_dataset_to_24k(data_dir: Path):
    src_dir = data_dir / "WAV"
    dst_dir = data_dir / "WAV_24"
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = list(src_dir.glob("*.wav"))
    print(f">>> Pre-processing {len(files)} files to 24kHz...")

    for wav_path in tqdm(files, desc="Resampling"):
        # 1. Load Original (Likely 16k)
        waveform, sr = torchaudio.load(wav_path)

        # 2. Resample to 24k
        if sr != SAMPLE_RATE_TARGET:  # 24000
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=SAMPLE_RATE_TARGET
            )
            waveform_24k = resampler(waveform)
        else:
            waveform_24k = waveform

        # 3. Save to WAV_24 (Same filename)
        save_path = dst_dir / wav_path.name
        torchaudio.save(save_path, waveform_24k, SAMPLE_RATE_TARGET)

    print(f">>> Completed! 24kHz files saved in {dst_dir}")


def parse_aligned_script(txt_path: Path, tokenizer) -> list[dict]:
    events = []

    pattern = re.compile(r"\[(\d+\.\d+),\s*(\d+\.\d+)\]\s+\S+\s+\S+\s+(.*)")

    if not txt_path.exists():
        return []

    IGNORE_TAGS = {
        "[*]",  # 판독 불가
        "[NPS]",  # 제3자 목소리
        "[PII]",  # 개인정보 (이름 등)
        "[SONANT]",  # 기침, 헛기침 등 생리적 소음
        "[MUSIC]",  # 음악/흥얼거림
        "[SYSTEM]",  # 기계음
        "[ENS]",  # 환경 소음
    }
    with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    start, end, content = match.groups()
                    start_t = float(start)
                    end_t = float(end)
                    content = content.strip()

                    if content in IGNORE_TAGS or not content: continue
                    
                    # [Optimization] 미리 토크나이징해서 ID 저장
                    token_ids = tokenizer.encode(content, add_special_tokens=False)

                    events.append({
                        "start": start_t,
                        "end": end_t,
                        "text": content,
                        "input_ids": token_ids, # [NEW] 저장됨
                        "duration": end_t - start_t,
                    })

    """
    {
        "start": 0.315,
        "end": 0.867,
        "text": "[SONANT]", 
        "input_ids": [....],  # 토크나이징된 ID 리스트
        "duration": 0.552
    },
    {
        "start": 3.200,
        "end": 5.320,
        "text": "ah hello J P how are you today?",
        "duration": 2.12
    },
    {
        "start": 9.920,
        "end": 10.900,
        "text": "yeah um.",
        .......
    """
    return sorted(events, key=lambda x: x["start"])




def ensure_mono_and_length(audio_chunk: np.ndarray, target_length: int) -> np.ndarray:
    # 1. Mono 변환 (2채널 이상일 경우 평균)
    if audio_chunk.ndim > 1:
        audio_chunk = np.mean(audio_chunk, axis=1)

    # 2. 길이 맞춤
    current_len = len(audio_chunk)
    if current_len == target_length:
        return audio_chunk.astype(np.float32)
    elif current_len < target_length:
        # 0.16*25000 보다 모자르면 뒤에 0 채우기
        pad_width = target_length - current_len
        return np.pad(audio_chunk, (0, pad_width), mode="constant").astype(np.float32)
    else:
        # 넘치면 자르기
        return audio_chunk[:target_length].astype(np.float32)


def create_duplex_dataset(data_dir: Path, model_path: str) -> DatasetDict:
    wav_dir_16k = data_dir / "WAV"
    wav_dir_24k = data_dir / "WAV_24"
    txt_dir = data_dir / "TXT"

    if not wav_dir_24k.exists() or not list(wav_dir_24k.glob("*.wav")):
        print(">>> WAV_24 folder not found. Running pre-processing first...")
        preprocess_dataset_to_24k(data_dir)
    
    print(f">>> Loading Tokenizer from {model_path} for pre-processing...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    sessions = {}
    for wav_file in wav_dir_16k.glob("*.wav"):
        parts = wav_file.stem.split("_")
        if len(parts) < 2: continue
        group_key = "_".join(parts[:-1])
        spk_id = parts[-1]
        if group_key not in sessions: sessions[group_key] = []
        sessions[group_key].append({
            "spk_id": spk_id, "wav_path_16k": wav_file,
            "wav_path_24k": wav_dir_24k / wav_file.name,
            "txt_path": txt_dir / f"{wav_file.stem}.txt",
        })

    def storage_generator():
        # [중요] 여기서 시간이 좀 걸리더라도 미리 다 계산합니다.
        for group_key, speakers in tqdm(sessions.items(), desc="Processing Storage & Embeddings"):
            if len(speakers) < 2: continue
            pairs = [(speakers[0], speakers[1]), (speakers[1], speakers[0])]
            for user_info, target_info in pairs:
                with open(user_info["wav_path_16k"], "rb") as f: u_bytes = f.read()
                with open(target_info["wav_path_24k"], "rb") as f: t_bytes = f.read()
                
                events = parse_aligned_script(target_info["txt_path"], tokenizer)
                
                # [핵심 수정] 기존 extract_speaker_embedding 함수 사용
                # utils.py에 정의된 대로 bytes를 넘김
                try:
                    spk_emb = extract_speaker_embedding(t_bytes, sample_rate=24000)
                except Exception as e:
                    print(f"[Warning] Failed to extract embedding for {user_info['wav_path_16k']}: {e}")
                    spk_emb = np.zeros(SPEAKER_EMBEDDING_DIM, dtype=np.float32)

                yield {
                    "session_id": f"{group_key}_{target_info['spk_id']}",
                    "user_audio": {"bytes": u_bytes, "path": None},
                    "target_audio": {"bytes": t_bytes, "path": None},
                    "events_json": json.dumps(events),
                    "speaker_embedding": spk_emb, # 저장!
                }

    def train_generator():
        for group_key, speakers in sessions.items():
            if len(speakers) < 2: continue
            pairs = [(speakers[0], speakers[1]), (speakers[1], speakers[0])]
            for user_info, target_info in pairs:
                sess_id = f"{group_key}_{target_info['spk_id']}"
                with sf.SoundFile(user_info["wav_path_16k"]) as f:
                    max_len = len(f)
                yield {
                    "session_id": sess_id, 
                    "seq_id": 0,
                    "start_sample": 0, 
                    "end_sample": max_len,
                }
    
    # [Feature 수정] speaker_embedding 필드 추가
    storage_features = Features({
        "session_id": Value("string"), 
        "user_audio": HFAudio(decode=False),
        "target_audio": HFAudio(decode=False), 
        "events_json": Value("string"),
        "speaker_embedding": Sequence(Value("float32"), length=SPEAKER_EMBEDDING_DIM) # 192차원
    })
    
    train_features = Features({
        "session_id": Value("string"), 
        "seq_id": Value("int32"),
        "start_sample": Value("int64"), 
        "end_sample": Value("int64"),
    })
    
    ds_storage = Dataset.from_generator(storage_generator, features=storage_features)
    ds_train = Dataset.from_generator(train_generator, features=train_features)
    return DatasetDict({"storage": ds_storage, "train": ds_train})

class DuplexTransform:
    def __init__(self, storage_dataset, config: DuplexConfig):
        self.storage = storage_dataset
        self.config = config
        self.id_to_idx = {sid: i for i, sid in enumerate(storage_dataset["session_id"])}
        self.chunk_samples_user = int(CHUNK_DURATION * SAMPLE_RATE_USER)
        self.chunk_samples_target = int(CHUNK_DURATION * SAMPLE_RATE_TARGET)

        # Tokenizer는 Padding ID 확인 등을 위해 로드는 필요함 (토크나이징은 안 함)
        print(f">>> Loading Processor from {config.model_path} for Pad ID check...")
        try:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(config.model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else config.silence_token_id
            
            # [NEW] 시스템 프롬프트 토크나이징 (한번만)
            self.system_prompt_ids = self.tokenizer.encode(DEFAULT_SYSTEM_PROMPT, add_special_tokens=False)
            
        except Exception as e:
            print(f"[Warning] Failed to load tokenizer: {e}")
            raise e

    def __call__(self, batch):
        out_dataset_rows = []
        batch_ids = batch["session_id"]
        
        for i in range(len(batch_ids)):
            sess_id = batch_ids[i]
            store_idx = self.id_to_idx[sess_id]
            store_row = self.storage[store_idx]

            u_bytes = store_row["user_audio"]["bytes"]
            t_bytes_24k = store_row["target_audio"]["bytes"]
            target_events = json.loads(store_row["events_json"])

            speaker_embedding = np.array(store_row["speaker_embedding"], dtype=np.float32)

            input_sequence = list(self.system_prompt_ids)
            input_audios_list: list[Audio] = []
            
            # Key: event_index, Value: list of indices in input_sequence
            event_token_map: dict[int, list[int]] = {}
            
            # [핵심] 텍스트 토큰 대기열 (Queue)
            # 구조: (token_id, event_index) 튜플을 저장
            token_queue = []
            
            # 현재 처리 중인 이벤트 포인터
            next_event_idx = 0
            
            with sf.SoundFile(io.BytesIO(u_bytes)) as f:
                u_full = f.read(dtype="float32")
            if u_full.ndim > 1: u_full = np.mean(u_full, axis=1)

            num_chunks = len(u_full) // self.chunk_samples_user
            
            for c in range(num_chunks):
                # =========================================================
                # [Step A] User Audio (4 tokens) - 시계 역할
                # =========================================================
                c_start_sec = c * CHUNK_DURATION
                c_end_sec = c_start_sec + CHUNK_DURATION
                
                idx_s = c * self.chunk_samples_user
                idx_e = idx_s + self.chunk_samples_user
                u_chunk = ensure_mono_and_length(u_full[idx_s:idx_e], self.chunk_samples_user)
                
                input_audios_list.append(Audio(waveform=u_chunk, sampling_rate=SAMPLE_RATE_USER))
                input_sequence.extend([self.config.audio_placeholder_token] * self.config.audio_token_ratio)

                # =========================================================
                # [Step B] Event Fetching (대기열 채우기)
                # =========================================================
                # 현재 시간(c_end_sec) 이전에 시작된 이벤트가 있다면 큐에 추가
                # (이미 큐에 넣은 이벤트는 next_event_idx로 건너뜀)
                while next_event_idx < len(target_events):
                    evt = target_events[next_event_idx]
                    
                    # 이벤트 시작 시간이 현재 청크 범위 안(또는 이전)에 들어왔다면 트리거
                    # 약간의 오차를 허용하기 위해 <= c_end_sec 사용
                    if evt["start"] < c_end_sec:
                        if "input_ids" in evt:
                            # (토큰ID, 이벤트인덱스) 형태로 저장 -> 나중에 오디오 자를 때 씀
                            for tid in evt["input_ids"]:
                                token_queue.append((tid, next_event_idx))
                        next_event_idx += 1
                    else:
                        # 아직 시간이 안 된 이벤트면 중단
                        break
                # =========================================================
                # [Step C] Text Token Emission (가변 길이 로직)
                # =========================================================
                # Case 1: 큐가 비었음 -> [Silence] (1개)
                # Case 2: 큐에 1개 남음 -> [Token, Pad] (2개)
                # Case 3: 큐에 2개 이상 -> [Token, Token] (2개)
                
                emit_tokens = []
                emit_event_idxs = []

                if not token_queue:
                    # 1. 큐가 비었음 -> 침묵 1개
                    emit_tokens.append(self.config.silence_token_id)
                    emit_event_idxs.append(None)
                
                elif len(token_queue) == 1:
                    # 2. 큐에 1개 남음 -> 텍스트 1개 + Pad 1개
                    tid, e_idx = token_queue.pop(0)
                    emit_tokens.append(tid)
                    emit_event_idxs.append(e_idx)
                    
                    # Pad 추가
                    emit_tokens.append(self.pad_token_id)
                    emit_event_idxs.append(None)
                    
                else:
                    # 3. 큐에 2개 이상 -> 텍스트 2개
                    for _ in range(2):
                        tid, e_idx = token_queue.pop(0)
                        emit_tokens.append(tid)
                        emit_event_idxs.append(e_idx)

                # 시퀀스 추가 및 인덱스 매핑
                start_pos = len(input_sequence)
                input_sequence.extend(emit_tokens)
                
                for k, e_idx in enumerate(emit_event_idxs):
                    if e_idx is not None:
                        if e_idx not in event_token_map:
                            event_token_map[e_idx] = []
                        event_token_map[e_idx].append(start_pos + k)

                if len(input_sequence) > self.config.max_token_length:
                    print(f"[Warning] Truncating sequence at {len(input_sequence)} tokens.")
                    break

            # --- [Step D] Target Audio Segments 생성 ---
            target_audios_list: list[AudioSeg] = []
            
            with sf.SoundFile(io.BytesIO(t_bytes_24k)) as f:
                full_target_audio = f.read(dtype="float32")
                if full_target_audio.ndim > 1:
                    full_target_audio = np.mean(full_target_audio, axis=1)

            for e_idx, indices in sorted(event_token_map.items()):
                evt = target_events[e_idx]
                t_start_sample = int(evt["start"] * SAMPLE_RATE_TARGET)
                t_end_sample = int(evt["end"] * SAMPLE_RATE_TARGET)
                
                if t_start_sample < 0: t_start_sample = 0
                if t_end_sample > len(full_target_audio): t_end_sample = len(full_target_audio)
                
                if t_end_sample > t_start_sample:
                    t_chunk = full_target_audio[t_start_sample:t_end_sample]
                else:
                    t_chunk = np.zeros(16000, dtype=np.float32)

                target_audios_list.append(AudioSeg(
                    text_token_idxs=indices,
                    audio=Audio(waveform=t_chunk, sampling_rate=SAMPLE_RATE_TARGET)
                ))

            row_obj = DatasetRow(
                input_sequence=input_sequence,
                target_audios=target_audios_list,
                input_audios=input_audios_list,
                speaker_embedding=speaker_embedding
            )
            out_dataset_rows.append(row_obj)
            
        return {"dataset_row_obj": out_dataset_rows}

def duplex_data(
    data_dir: Optional[Path] = None,
    cache_dir: Path = Path("./dataset_duplex"),
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> Dataset:
    dataset_path = cache_dir
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar_path = dataset_path.parent / "temp_cache.tar"

    if not dataset_path.exists():
        if data_dir is not None and data_dir.exists():
            print(f">>> Creating dataset from raw data at {data_dir}...")
            dataset = create_duplex_dataset(data_dir, model_path) 
            dataset.save_to_disk(str(dataset_path))
        else:
            print(
                ">>> Raw data not provided. Fetching config from GitHub for Duplex..."
            )
            url_url = "https://raw.githubusercontent.com/riverfog7/sca_data_prep/refs/heads/main/.hf_dataset_url_duplex"
            hash_url = "https://raw.githubusercontent.com/riverfog7/sca_data_prep/refs/heads/main/.hf_dataset_md5_duplex"

            # url_url = "https://raw.githubusercontent.com/wjm9765/sca_data_prep/refs/heads/main/.hf_dataset_url_duplex"
            # hash_url = "https://raw.githubusercontent.com/wjm9765/sca_data_prep/refs/heads/main/.hf_dataset_md5_duplex"

            try:
                print(f"Reading URL from {url_url}...")
                dataset_url = requests.get(url_url).text.strip()

                print(f"Reading MD5 from {hash_url}...")
                dataset_md5 = requests.get(hash_url).text.strip()

                print(f"Target URL: {dataset_url}")

                hash_func = md5()
                dl_stream = requests.get(dataset_url, stream=True)
                dl_stream.raise_for_status()
                total_size = int(dl_stream.headers.get("content-length", 0))

                with (
                    open(tmp_tar_path, "wb") as f,
                    tqdm(
                        desc="Downloading dataset",
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar,
                ):
                    for chunk in dl_stream.iter_content(chunk_size=8192):
                        f.write(chunk)
                        hash_func.update(chunk)
                        bar.update(len(chunk))

                if hash_func.hexdigest() != dataset_md5:
                    shutil.rmtree(dataset_path, ignore_errors=True)
                    tmp_tar_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"Downloaded dataset file is corrupted (MD5 mismatch)\nExpected: {dataset_md5}\nGot: {hash_func.hexdigest()}"
                    )

                print(f">>> Extracting to {dataset_path.parent}...")
                with tarfile.open(tmp_tar_path, "r") as tar:
                    tar.extractall(path=dataset_path.parent)

            except Exception as e:
                print(f"[Error] Download failed: {e}")
                if tmp_tar_path.exists():
                    tmp_tar_path.unlink()
                if dataset_path.exists():
                    shutil.rmtree(dataset_path, ignore_errors=True)
                raise e
            finally:
                if tmp_tar_path.exists():
                    tmp_tar_path.unlink()

    print(f">>> Loading dataset from disk: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    assert isinstance(dataset, DatasetDict)
    train_ds = dataset["train"]

    print(">>> Setting up DuplexTransform...")
    
    config = DuplexConfig(model_path=model_path)
    train_ds.set_transform(DuplexTransform(dataset["storage"], config=config))
 

    return train_ds


def remove_extras(
    session: ComedySession,
    remove_events: Tuple[type[BaseEvent], ...] = (AudienceEvent, EnvironmentEvent),
) -> ComedySession:
    filtered = []
    for event in session.timeline:
        if not isinstance(event, remove_events):
            filtered.append(event)

    return ComedySession(timeline=filtered, video_id=session.video_id)


def assert_overlap(session: ComedySession) -> None:
    overlap_threshold = 0.00001
    for i, event in enumerate(session.timeline):
        if len(session.timeline) - 1 == i:
            break
        next_event = session.timeline[i + 1]

        assert next_event.start + overlap_threshold >= event.end, (
            f"Events overlap: {event} and {next_event}"
        )


def merge_close_events(
    session: ComedySession, gap_threshold: float = 0.5
) -> ComedySession:
    if not session.timeline:
        return session

    merged_timeline = []
    current_evt = session.timeline[0]

    for i in range(1, len(session.timeline)):
        next_evt = session.timeline[i]

        gap = next_evt.start - current_evt.end
        if gap < gap_threshold:
            current_evt = ComedianEvent(
                start=current_evt.start,
                end=next_evt.end,
                content=f"{current_evt.content}{next_evt.content}",
                event_type=current_evt.event_type,
                role="comedian",
                delivery_tag=None,
            )
        else:
            merged_timeline.append(current_evt)
            current_evt = next_evt

    merged_timeline.append(current_evt)

    return ComedySession(timeline=merged_timeline, video_id=session.video_id)


def to_hf_dataset(
    sessions: Iterable[ComedySession],
    audio_base_path: Path,
    min_duration: float,
    max_duration: float,
    cut_start: int,
    cut_end: int,
    min_speech_duration: float,
) -> DatasetDict:
    event_rows = []
    unique_sessions = {}
    for session in sessions:
        if session.video_id not in unique_sessions:
            audio_path = list(audio_base_path.glob(f"{session.video_id}.*"))
            if len(audio_path) != 1:
                raise FileNotFoundError(
                    f"Audio file not found for session {session.video_id} in {audio_base_path}"
                )
            unique_sessions[session.video_id] = str(audio_path[0])

        for i, event in enumerate(session.timeline):
            if i < cut_start or i >= len(session.timeline) - cut_end:
                continue
            if isinstance(event, ComedianEvent) and event.event_type == "speech":
                # Don't include very short or very long segments
                if event.start < min_duration or event.start > max_duration:
                    continue
                if (event.end - event.start) < min_speech_duration:
                    continue
                event_rows.append(
                    {
                        "session_id": session.video_id,
                        # (A) Input Context: 0.0 ~ 현재 대사 시작 전 (원본 오디오)
                        "start_sec": 0.0,
                        "end_sec": event.start,
                        # (B) Target Audio: 현재 대사 시작 ~ 끝 (Clean 오디오)
                        "target_start_sec": event.start,
                        "target_end_sec": event.end,
                        "target_text": event.content,
                        "event_index": i,
                    }
                )
            else:
                raise ValueError(
                    f"Unexpected event type in session {session.video_id}: {event}"
                )

    def audio_generator():
        for sess_id, path in tqdm(unique_sessions.items(), desc="Processing Audio"):
            try:
                with open(path, "rb") as f:
                    original_bytes = f.read()

                # Moshi's mimi neural audio codec takes 24kHz input
                print(f"[{sess_id}] Starting audio cleaning...")
                cleaned_bytes = clean_audio_bytes(original_bytes, target_sr=24000)
                print(f"[{sess_id}] Cleaned audio successfully.")

                # Extract speaker embedding from cleaned audio (trims 30s from start/end)
                print(f"[{sess_id}] Extracting speaker embedding...")
                speaker_embedding = extract_speaker_embedding(
                    cleaned_bytes, sample_rate=24000
                )
                print(
                    f"[{sess_id}] Extracted speaker embedding: shape {speaker_embedding.shape}"
                )

                yield {
                    "session_id": sess_id,
                    "audio": {
                        "bytes": check_and_resample_audio(
                            original_bytes, target_sr=16000
                        )
                    },  # 원본 (Context용)
                    "clean_audio": {"bytes": cleaned_bytes},  # AI 처리됨 (Target용)
                    "speaker_embedding": speaker_embedding,  # [192] ECAPA-TDNN embedding
                }
            except Exception as e:
                print(f"\n{'=' * 80}")
                print(f"ERROR processing session: {sess_id}")
                print(f"Audio file: {path}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"{'=' * 80}\n")
                raise

    text_features = Features(
        {
            "session_id": Value("string"),
            "start_sec": Value("float"),
            "end_sec": Value("float"),
            "target_start_sec": Value("float"),
            "target_end_sec": Value("float"),
            "target_text": Value("string"),
            "event_index": Value("int32"),
        }
    )

    audio_features = Features(
        {
            "session_id": Value("string"),
            "audio": HFAudio(decode=False),
            "clean_audio": HFAudio(decode=False),
            "speaker_embedding": Sequence(
                Value("float32"), length=SPEAKER_EMBEDDING_DIM
            ),  # [192] ECAPA-TDNN
        }
    )

    ds_text = Dataset.from_list(event_rows, features=text_features)
    ds_audio = Dataset.from_generator(audio_generator, features=audio_features)

    return DatasetDict(
        {
            "storage": ds_audio,
            "train": ds_text,
        }
    )


def to_talker_chat_format_batch(
    batch: dict,
    system_prompt: Optional[str] = None,
    instruction_prompt: Optional[str] = None,
) -> dict:
    messages_list = []

    # batch['audio'] -> Input Context (Original)
    # batch['target_audio'] -> Target Speech (Clean)
    # batch['target_text'] -> Target Text
    # batch['speaker_embedding'] -> Pre-computed speaker embedding [192]

    for input_audio, target_audio, text, speaker_emb in zip(
        batch["audio"],
        batch["target_audio"],
        batch["target_text"],
        batch["speaker_embedding"],
    ):
        msgs = [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_waveform": input_audio["array"],
                        "sampling_rate": input_audio["sampling_rate"],
                    },
                    {
                        "type": "text",
                        "text": instruction_prompt or DEFAULT_INSTRUCTION_PROMPT,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text.strip()},
                    {
                        "type": "audio",
                        "audio_waveform": target_audio["array"],
                        "sampling_rate": target_audio["sampling_rate"],
                        "speaker_embedding": speaker_emb,  # [192] ECAPA-TDNN embedding
                    },
                ],
            },
        ]
        messages_list.append(msgs)

    return {"messages": messages_list}


def to_chat_format(
    row, system_prompt: Optional[str] = None, instruction_prompt: Optional[str] = None
) -> dict:
    audio_data = row["audio"]["array"]
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_waveform": audio_data,
                    "sampling_rate": row["audio"]["sampling_rate"],
                },
                {
                    "type": "text",
                    "text": instruction_prompt or DEFAULT_INSTRUCTION_PROMPT,
                },
            ],
        },
        {"role": "assistant", "content": row["target_text"].strip()},
    ]
    return {"messages": messages}


def to_chat_format_batch(
    batch: dict,
    system_prompt: Optional[str] = None,
    instruction_prompt: Optional[str] = None,
) -> dict:
    messages_list = []
    for audio_entry, target_text in zip(batch["audio"], batch["target_text"]):
        fake_row = {"audio": audio_entry, "target_text": target_text}
        result = to_chat_format(fake_row, system_prompt, instruction_prompt)
        messages_list.append(result["messages"])

    return {"messages": messages_list}


def easy_load(
    dataset_path: Optional[Path] = None,
    cache_dir: Path = Path("./dataset"),
    format: Literal["chat", "raw", "talker_chat", "duplex"] = "talker_chat",
    system_prompt: Optional[str] = None,
    instruction_prompt: Optional[str] = None,
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> Dataset:
    if format == "duplex":
        return duplex_data(dataset_path, cache_dir, model_path=model_path)

    if dataset_path is None:
        dataset_path = cache_dir / "sca_comedy_dataset"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_tar_path = dataset_path.parent / "sca_comedy_dataset.tar"

        if not dataset_path.exists():
            url_url = "https://raw.githubusercontent.com/riverfog7/sca_data_prep/refs/heads/main/.hf_dataset_url"
            hash_url = "https://raw.githubusercontent.com/riverfog7/sca_data_prep/refs/heads/main/.hf_dataset_md5"
            dataset_url = requests.get(url_url).text.strip()
            dataset_md5 = requests.get(hash_url).text.strip()

            hash_func = md5()
            dl_stream = requests.get(dataset_url, stream=True)
            total_size = int(dl_stream.headers.get("content-length", 0))

            with (
                open(tmp_tar_path, "wb") as f,
                tqdm(
                    desc="Downloading dataset",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for chunk in dl_stream.iter_content(chunk_size=8192):
                    f.write(chunk)
                    hash_func.update(chunk)
                    bar.update(len(chunk))

            if hash_func.hexdigest() != dataset_md5:
                shutil.rmtree(dataset_path, ignore_errors=True)
                tmp_tar_path.unlink(missing_ok=True)
                raise ValueError("Downloaded dataset file is corrupted (MD5 mismatch)")

            with tarfile.open(tmp_tar_path, "r") as tar:
                tar.extractall(path=dataset_path.parent)

            tmp_tar_path.unlink(missing_ok=True)

    dataset = load_from_disk(dataset_path)
    assert isinstance(dataset, DatasetDict), (
        "Loaded dataset is not a DatasetDict. Something is corrupted."
    )
    train_ds = dataset["train"]

    if format == "chat":
        loader = RelationalAudioLoader(dataset["storage"])
        train_ds.set_transform(
            lambda batch: to_chat_format_batch(
                loader(batch), system_prompt, instruction_prompt
            )
        )
    elif format == "talker_chat":
        loader = TalkerAudioLoader(dataset["storage"])
        train_ds.set_transform(
            lambda batch: to_talker_chat_format_batch(
                loader(batch), system_prompt, instruction_prompt
            )
        )
    elif format == "raw":
        loader = RelationalAudioLoader(dataset["storage"])
        train_ds.set_transform(loader)
    else:
        raise ValueError(f"Unsupported format: {format}")
    return train_ds


class RelationalAudioLoader:
    def __init__(self, audio_dataset):
        self.audio_dataset = audio_dataset
        self.id_to_idx = {
            sess_id: idx for idx, sess_id in enumerate(audio_dataset["session_id"])
        }

    def __call__(self, batch):
        audio_arrays = []
        sampling_rates = []

        for session_id, start, end in zip(
            batch["session_id"], batch["start_sec"], batch["end_sec"]
        ):
            try:
                row_idx = self.id_to_idx.get(session_id)
                if row_idx is None:
                    raise ValueError(f"Session {session_id} not found")

                audio_entry = self.audio_dataset[row_idx]["audio"]
                raw_bytes = audio_entry["bytes"]

                with io.BytesIO(raw_bytes) as file_obj:
                    with sf.SoundFile(file_obj) as f:
                        sr = f.samplerate
                        start_frame = int(start * sr)
                        frames_to_read = int((end - start) * sr)

                        if frames_to_read <= 0:
                            audio_arrays.append(np.array([0.0], dtype=np.float32))
                            sampling_rates.append(sr)
                            continue

                        f.seek(start_frame)
                        y = f.read(frames=frames_to_read, dtype="float32")
                        if y.ndim > 1:
                            y = y.mean(axis=1)

                        audio_arrays.append(y)
                        sampling_rates.append(sr)

            except Exception as e:
                print(f"Error: {e}")
                audio_arrays.append(np.array([0.0], dtype=np.float32))
                sampling_rates.append(16000)

        batch["audio"] = [
            {"array": arr, "sampling_rate": sr}
            for arr, sr in zip(audio_arrays, sampling_rates)
        ]
        return batch


class TalkerAudioLoader(RelationalAudioLoader):
    def __call__(self, batch):
        batch = super().__call__(batch)

        target_arrays = []
        target_srs = []
        speaker_embeddings = []

        for session_id, t_start, t_end in zip(
            batch["session_id"], batch["target_start_sec"], batch["target_end_sec"]
        ):
            try:
                row_idx = self.id_to_idx.get(session_id)

                clean_entry = self.audio_dataset[row_idx]["clean_audio"]
                raw_bytes = clean_entry["bytes"]

                with io.BytesIO(raw_bytes) as file_obj:
                    with sf.SoundFile(file_obj) as f:
                        sr = f.samplerate
                        start_frame = int(t_start * sr)
                        frames_to_read = int((t_end - t_start) * sr)

                        if frames_to_read <= 0:
                            target_arrays.append(np.array([0.0], dtype=np.float32))
                            target_srs.append(sr)
                            speaker_embeddings.append(
                                self.audio_dataset[row_idx]["speaker_embedding"]
                            )
                            continue

                        f.seek(start_frame)
                        y = f.read(frames=frames_to_read, dtype="float32")
                        if y.ndim > 1:
                            y = y.mean(axis=1)

                        target_arrays.append(y)
                        target_srs.append(sr)

                # Load pre-computed speaker embedding for this session
                speaker_embeddings.append(
                    self.audio_dataset[row_idx]["speaker_embedding"]
                )

            except Exception as e:
                print(f"Target Audio Error: {e}")
                target_arrays.append(np.array([0.0], dtype=np.float32))
                target_srs.append(16000)
                # Re-raise since we don't have a valid speaker embedding fallback
                raise e

        batch["target_audio"] = [
            {"array": arr, "sampling_rate": sr}
            for arr, sr in zip(target_arrays, target_srs)
        ]
        batch["speaker_embedding"] = speaker_embeddings
        return batch
