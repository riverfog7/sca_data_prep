import io
import shutil
import tarfile
from hashlib import md5
from pathlib import Path
from typing import Tuple, Optional, Iterable, Literal

import numpy as np
import requests
import soundfile as sf
from datasets import DatasetDict, Dataset, load_from_disk
from datasets import Features, Value, Audio, Sequence
from tqdm import tqdm
import re 
import math

from .constants import DEFAULT_SYSTEM_PROMPT, DEFAULT_INSTRUCTION_PROMPT
from .models.events import ComedianEvent, BaseEvent, AudienceEvent, EnvironmentEvent, ComedySession
from .utils import clean_audio_bytes, check_and_resample_audio, extract_speaker_embedding, SPEAKER_EMBEDDING_DIM


# 설정: 0.16초 청크
CHUNK_DURATION = 0.16  
# 시퀀스 길이: 15분 (900초)
SEQUENCE_DURATION = 900 
#슬라이딩 윈도우 간격 3분
SEQUENCE_STRIDE = 180  
SAMPLE_RATE = 16000

def parse_aligned_script(txt_path:Path) -> list[dict]:
    events=[]

    pattern = re.compile(r'\[(\d+\.\d+),\s*(\d+\.\d+)\]\s+\S+\s+\S+\s+(.*)')
    
    if not txt_path.exists():
        return []


    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                start, end, content = match.groups()
                start_t = float(start)
                end_t = float(end)
                content = content.strip()
                
                #단어를 쪼갬 (why? 0.16초 후 5초 분량의 텍스트를 예측해야 한다면 지연시간 증가)
                words = content.split()
                
                events.append({
                    "start": start_t,
                    "end": end_t,
                    "text": content,
                    "words": words,               
                    "duration": end_t - start_t   
                })
    
    """
    {
        "start": 0.315,
        "end": 0.867,
        "text": "[SONANT]"
        "words": [],
        "duration": 0.552
    },
    {
        "start": 3.200,
        "end": 5.320,
        "text": "ah hello J P how are you today?",
        "words": ["ah", "hello", "J", "P", "how", "are", "you", "today?"],
        "duration": 2.12
    },
    {
        "start": 9.920,
        "end": 10.900,
        "text": "yeah um."
        .......
    """
    return sorted(events, key=lambda x: x["start"])


def get_sliced_text(chunk_start: float, chunk_end: float, events: list[dict]):
    sliced_text = ""
    is_speech = False
    
    for evt in events:
        overlap_start = max(chunk_start, evt["start"])
        overlap_end = min(chunk_end, evt["end"])
        
        if overlap_end > overlap_start:
            is_speech = True
            
            if evt["duration"] > 0 and evt["words"]:
                rel_start = (overlap_start - evt["start"]) / evt["duration"]
                rel_end = (overlap_end - evt["start"]) / evt["duration"]
                
                n_words = len(evt["words"])
                
                w_start = int(rel_start * n_words)
                w_end = int(math.ceil(rel_end * n_words)) 
                
                w_start = max(0, w_start)
                w_end = min(n_words, w_end)
                
                current_words = evt["words"][w_start:w_end]
                if current_words:
                    sliced_text = " ".join(current_words)
                    
            # 0.16초는 매우 짧으므로, 한 이벤트 안에 여러 청크가 들어감.
            # 루프는 계속 돌되, 현재 청크 범위를 벗어난 이벤트는 볼 필요 없음
        
        if evt["start"] > chunk_end:
            break
            
    return is_speech, sliced_text


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
        return np.pad(audio_chunk, (0, pad_width), mode='constant').astype(np.float32)
    else:
        # 넘치면 자르기
        return audio_chunk[:target_length].astype(np.float32)

def create_duplex_dataset(data_dir: Path) -> DatasetDict:
    wav_dir = data_dir / "WAV"
    txt_dir = data_dir / "TXT"
    
    sessions = {}
    for wav_file in wav_dir.glob("*.wav"):
        parts = wav_file.stem.split('_')
        if len(parts) < 2: continue
        group_key = "_".join(parts[:-1])
        spk_id = parts[-1]
        if group_key not in sessions: sessions[group_key] = []
        sessions[group_key].append({"spk_id": spk_id, "wav_path": wav_file, "txt_path": txt_dir / f"{wav_file.stem}.txt"})
    def storage_generator():
        for group_key, speakers in tqdm(sessions.items(), desc="Processing Storage"):
            if len(speakers) < 2: continue
            pairs = [(speakers[0], speakers[1]), (speakers[1], speakers[0])]
            
            for user_info, target_info in pairs:
                # 오디오 바이트 읽기
                with open(user_info["wav_path"], "rb") as f: u_bytes = f.read()
                with open(target_info["wav_path"], "rb") as f: t_bytes = f.read()
                
                yield {
                    "session_id": f"{group_key}_{target_info['spk_id']}",
                    "user_audio": {"bytes": u_bytes, "path": None},
                    "target_audio": {"bytes": t_bytes, "path": None},
                    "txt_path": str(target_info["txt_path"])
                }

    def train_generator():
        for group_key, speakers in sessions.items():
            if len(speakers) < 2: continue
            pairs = [(speakers[0], speakers[1]), (speakers[1], speakers[0])]
            for user_info, target_info in pairs:
                sess_id = f"{group_key}_{target_info['spk_id']}"
                with sf.SoundFile(user_info["wav_path"]) as f:
                    max_len = len(f)
                    sr = f.samplerate
                samples_per_seq = int(SEQUENCE_DURATION * sr)
                stride_samples = int(SEQUENCE_STRIDE * sr)
                for seq_idx, start_sample in enumerate(range(0, max_len, stride_samples)):
                    end_sample = min(start_sample + samples_per_seq, max_len)
                    if end_sample <= start_sample: break
                    yield {
                        "session_id": sess_id,
                        "seq_id": seq_idx,
                        "start_sample": start_sample,
                        "end_sample": end_sample
                    }

    storage_features = Features({
        "session_id": Value("string"),
        "user_audio": Audio(decode=False),   
        "target_audio": Audio(decode=False), 
        "txt_path": Value("string")
    })
    
    train_features = Features({
        "session_id": Value("string"),
        "seq_id": Value("int32"),
        "start_sample": Value("int64"),
        "end_sample": Value("int64")
    })

    ds_storage = Dataset.from_generator(storage_generator, features=storage_features)
    ds_train = Dataset.from_generator(train_generator, features=train_features)
    
    return DatasetDict({"storage": ds_storage, "train": ds_train})

class DuplexTransform:
    def __init__(self, storage_dataset, sample_rate=16000):
        self.storage = storage_dataset
        self.sample_rate = sample_rate
        self.chunk_samples = int(CHUNK_DURATION * sample_rate)
        self.id_to_idx = {sid: i for i, sid in enumerate(storage_dataset["session_id"])}

    def __call__(self, batch):
        batch_size = len(batch["session_id"])
        out_types, out_waveforms, out_texts, out_labels = [], [], [], []
        
        for i in range(batch_size):
            sess_id = batch["session_id"][i]
            start = batch["start_sample"][i]
            end = batch["end_sample"][i]
            
            store_idx = self.id_to_idx[sess_id]
            store_row = self.storage[store_idx]
            
            u_bytes = store_row["user_audio"]["bytes"]
            t_bytes = store_row["target_audio"]["bytes"]
            
            with sf.SoundFile(io.BytesIO(u_bytes)) as f:
                f.seek(start)
                u_seq = f.read(end - start)
            with sf.SoundFile(io.BytesIO(t_bytes)) as f:
                f.seek(start)
                t_seq = f.read(end - start)
            
            curr_len = end - start
            if len(u_seq) < curr_len:
                pad = curr_len - len(u_seq)
                u_seq = np.pad(u_seq, (0, pad)) if u_seq.ndim==1 else np.pad(u_seq, ((0,pad),(0,0)))
                t_seq = np.pad(t_seq, (0, pad)) if t_seq.ndim==1 else np.pad(t_seq, ((0,pad),(0,0)))

            target_events = parse_aligned_script(Path(store_row["txt_path"]))
            chunk_count = curr_len // self.chunk_samples
            
            seq_types, seq_waves, seq_txts, seq_lbls = [], [], [], []
            for c in range(chunk_count):
                idx_s = c * self.chunk_samples
                idx_e = idx_s + self.chunk_samples
                u_chunk = ensure_mono_and_length(u_seq[idx_s:idx_e], self.chunk_samples)
                t_chunk = ensure_mono_and_length(t_seq[idx_s:idx_e], self.chunk_samples)
                c_start_sec = (start / self.sample_rate) + (c * CHUNK_DURATION)
                c_end_sec = c_start_sec + CHUNK_DURATION
                
                is_speech, text_slice = get_sliced_text(c_start_sec, c_end_sec, target_events)
                
                # A. User
                seq_types.append("user_audio")
                seq_waves.append(u_chunk)
                seq_txts.append("")
                seq_lbls.append(-100)
                # B. Text
                if text_slice:
                    seq_types.append("text")
                    seq_waves.append(np.zeros(self.chunk_samples, dtype=np.float32))
                    seq_txts.append(text_slice)
                    seq_lbls.append(1)
                # C. Target
                seq_types.append("target_audio")
                seq_waves.append(t_chunk)
                seq_txts.append("")
                seq_lbls.append(1)
            
            out_types.append(seq_types)
            out_waveforms.append(seq_waves)
            out_texts.append(seq_txts)
            out_labels.append(seq_lbls)
            
        batch["types"] = out_types
        batch["waveforms"] = out_waveforms
        batch["texts"] = out_texts
        batch["label_mask"] = out_labels
        return batch

def duplex_data(data_dir: Optional[Path] = None, cache_dir: Optional[Path] = Path('./dataset_duplex')) -> Dataset:
    DATASET_URL = "https://huggingface.co/datasets/wjm9765/sca_full_duplex/resolve/main/sca_duplex_cache.tar?download=true" 

    dataset_path = cache_dir  # easy_load의 dataset_path와 동일한 역할
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar_path = dataset_path.parent / "temp_cache.tar"

    if not dataset_path.exists():
        
        if DATASET_URL:
            print(f">>> Downloading dataset from {DATASET_URL}...")
            
            try:
                with requests.get(DATASET_URL, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    
                    with open(tmp_tar_path, "wb") as f, tqdm(
                            desc="Downloading dataset",
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            bar.update(len(chunk))

                print(f">>> Extracting to {dataset_path.parent}...")
                with tarfile.open(tmp_tar_path, "r") as tar:
                    tar.extractall(path=dataset_path.parent)
                
            except Exception as e:
                print(f"[Error] Download failed: {e}")
                if tmp_tar_path.exists(): tmp_tar_path.unlink()
                if dataset_path.exists(): shutil.rmtree(dataset_path)
                raise e
            finally:
                if tmp_tar_path.exists():
                    tmp_tar_path.unlink()

        elif data_dir is not None and data_dir.exists():
            print(f">>> Cache not found. Creating from raw data at {data_dir}...")
            dataset = create_duplex_dataset(data_dir)
            dataset.save_to_disk(str(dataset_path))
        
        else:
            raise FileNotFoundError(f"Dataset not found at {dataset_path} and no URL/RawData provided.")

    print(f">>> Loading dataset from disk: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    train_ds = dataset["train"]
    
    print(">>> Setting up DuplexTransform...")
    train_ds.set_transform(DuplexTransform(dataset["storage"], sample_rate=SAMPLE_RATE))
    
    return train_ds


# def duplex_data(data_dir: Path=None, cache_dir: str = "./dataset_duplex") -> Dataset:
#     cache_path = Path(cache_dir)

#     # 1. 캐시가 없으면 다운로드 시도
#     if not cache_path.exists():
#         print(f">>> Cache not found at {cache_path}")
        
#         # [수정 필요] ★ 여기에 아까 복사한 Hugging Face 다운로드 링크를 넣으세요! ★
#         CACHE_URL = "https://huggingface.co/datasets/YOUR_ID/REPO_NAME/resolve/main/sca_duplex_cache.tar"
        
#         try:
#             # data_dir이 제공되지 않았으면 다운로드 모드로 진입
#             download_and_extract(CACHE_URL, cache_path)
#         except Exception as e:
#             print(f">>> Auto-download failed: {e}")
#             if data_dir is not None and data_dir.exists():
#                 print(f">>> Trying to create from raw data at {data_dir}...")
#                 dataset = create_duplex_dataset(data_dir)
#                 dataset.save_to_disk(str(cache_path))
#             else:
#                 raise FileNotFoundError("No cache found and no raw data_dir provided.")

#     # 2. 로드
#     print(f">>> Loading dataset from cache: {cache_path}")
#     dataset = load_from_disk(str(cache_path))
    
#     # 3. Transform 설정
#     dataset["train"].set_transform(DuplexTransform(dataset["storage"], sample_rate=SAMPLE_RATE))
    
#     return dataset["train"]



def remove_extras(session: ComedySession, remove_events: Tuple[BaseEvent] = (AudienceEvent, EnvironmentEvent)) -> ComedySession:
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

        assert next_event.start + overlap_threshold >= event.end, f"Events overlap: {event} and {next_event}"


def merge_close_events(session: ComedySession, gap_threshold: float = 0.5) -> ComedySession:
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
                role='comedian',
                delivery_tag=None,
            )
        else:
            merged_timeline.append(current_evt)
            current_evt = next_evt

    merged_timeline.append(current_evt)

    return ComedySession(timeline=merged_timeline, video_id=session.video_id)


def to_hf_dataset(sessions: Iterable[ComedySession], audio_base_path: Path, min_duration: float, max_duration: float, cut_start: int, cut_end: int, min_speech_duration: float) -> DatasetDict:
    event_rows = []
    unique_sessions = {}
    for session in sessions:
        if session.video_id not in unique_sessions:
            audio_path = list(audio_base_path.glob(f"{session.video_id}.*"))
            if len(audio_path) != 1:
                raise FileNotFoundError(f"Audio file not found for session {session.video_id} in {audio_base_path}")
            unique_sessions[session.video_id] = str(audio_path[0])

        for i, event in enumerate(session.timeline):
            if i < cut_start or i >= len(session.timeline) - cut_end:
                continue
            if isinstance(event, ComedianEvent) and event.event_type == 'speech':
                # Don't include very short or very long segments
                if event.start < min_duration or event.start > max_duration:
                    continue
                if (event.end - event.start) < min_speech_duration:
                    continue
                event_rows.append({
                    "session_id": session.video_id,
                    # (A) Input Context: 0.0 ~ 현재 대사 시작 전 (원본 오디오)
                    "start_sec": 0.0,
                    "end_sec": event.start,
                    # (B) Target Audio: 현재 대사 시작 ~ 끝 (Clean 오디오)
                    "target_start_sec": event.start,
                    "target_end_sec": event.end,
                    
                    "target_text": event.content,
                    "event_index": i
                })
            else:
                raise ValueError(f"Unexpected event type in session {session.video_id}: {event}")




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
                speaker_embedding = extract_speaker_embedding(cleaned_bytes, sample_rate=24000)
                print(f"[{sess_id}] Extracted speaker embedding: shape {speaker_embedding.shape}")
                
                yield {
                    "session_id": sess_id,
                    "audio": {"bytes": check_and_resample_audio(original_bytes, target_sr=16000)},    # 원본 (Context용)
                    "clean_audio": {"bytes": cleaned_bytes},  # AI 처리됨 (Target용)
                    "speaker_embedding": speaker_embedding,   # [192] ECAPA-TDNN embedding
                }
            except Exception as e:
                print(f"\n{'='*80}")
                print(f"ERROR processing session: {sess_id}")
                print(f"Audio file: {path}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"{'='*80}\n")
                raise

    
    text_features = Features({
        "session_id": Value("string"),
        "start_sec": Value("float"),
        "end_sec": Value("float"),
        "target_start_sec": Value("float"), # 추가됨
        "target_end_sec": Value("float"),   # 추가됨
        "target_text": Value("string"),
        "event_index": Value("int32"),
    })
    
    audio_features = Features({
        "session_id": Value("string"),
        "audio": Audio(decode=False),       # 원본
        "clean_audio": Audio(decode=False),  # Clean 버전 추가
        "speaker_embedding": Sequence(Value("float32"), length=SPEAKER_EMBEDDING_DIM),  # [192] ECAPA-TDNN
    })

    ds_text = Dataset.from_list(event_rows, features=text_features)
    ds_audio = Dataset.from_generator(audio_generator, features=audio_features)

    return DatasetDict({
        "storage": ds_audio,
        "train": ds_text,
    })

def to_talker_chat_format_batch(batch: dict, system_prompt: Optional[str] = None, instruction_prompt: Optional[str] = None) -> dict:
    messages_list = []
    
    # batch['audio'] -> Input Context (Original)
    # batch['target_audio'] -> Target Speech (Clean)
    # batch['target_text'] -> Target Text
    # batch['speaker_embedding'] -> Pre-computed speaker embedding [192]
    
    for input_audio, target_audio, text, speaker_emb in zip(
        batch["audio"], batch["target_audio"], batch["target_text"], batch["speaker_embedding"]
    ):
        msgs = [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_waveform": input_audio["array"], "sampling_rate": input_audio["sampling_rate"]},
                    {"type": "text", "text": instruction_prompt or DEFAULT_INSTRUCTION_PROMPT},
                ]
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
                    }
                ]
            }
        ]
        messages_list.append(msgs)

    return {"messages": messages_list}

def to_chat_format(row, system_prompt: Optional[str] = None, instruction_prompt: Optional[str] = None) -> dict:
    audio_data = row["audio"]["array"]
    messages = [
        {
            "role": "system",
            "content": system_prompt or DEFAULT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_waveform": audio_data,
                    "sampling_rate": row["audio"]["sampling_rate"]
                },
                {"type": "text", "text": instruction_prompt or DEFAULT_INSTRUCTION_PROMPT},
            ]
        },
        {
            "role": "assistant",
            "content": row["target_text"].strip()
        }
    ]
    return {"messages": messages}


def to_chat_format_batch(batch: dict, system_prompt: Optional[str] = None, instruction_prompt: Optional[str] = None) -> dict:
    messages_list = []
    for audio_entry, target_text in zip(batch["audio"], batch["target_text"]):
        fake_row = {
            "audio": audio_entry,
            "target_text": target_text
        }
        result = to_chat_format(fake_row, system_prompt, instruction_prompt)
        messages_list.append(result["messages"])

    return {"messages": messages_list}


def easy_load(dataset_path: Optional[Path] = None, cache_dir: Optional[Path] = Path('./dataset'), format: Literal["chat", "raw", "talker_chat"] = "talker_chat", system_prompt: Optional[str] = None, instruction_prompt: Optional[str] = None) -> Dataset:
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
            total_size = int(dl_stream.headers.get('content-length', 0))

            with open(tmp_tar_path, "wb") as f, tqdm(
                    desc="Downloading dataset",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
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
    train_ds = dataset["train"]

    if format == "chat":
        loader = RelationalAudioLoader(dataset["storage"])
        train_ds.set_transform(lambda batch: to_chat_format_batch(loader(batch), system_prompt, instruction_prompt))
    elif format == "talker_chat":
        loader = TalkerAudioLoader(dataset["storage"])
        train_ds.set_transform(lambda batch: to_talker_chat_format_batch(loader(batch), system_prompt, instruction_prompt))
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
            sess_id: idx
            for idx, sess_id in enumerate(audio_dataset["session_id"])
        }

    def __call__(self, batch):
        audio_arrays = []
        sampling_rates = []

        for session_id, start, end in zip(batch['session_id'], batch['start_sec'], batch['end_sec']):
            try:
                row_idx = self.id_to_idx.get(session_id)
                if row_idx is None:
                    raise ValueError(f"Session {session_id} not found")

                audio_entry = self.audio_dataset[row_idx]["audio"]
                raw_bytes = audio_entry['bytes']

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
                        y = f.read(frames=frames_to_read, dtype='float32')
                        if y.ndim > 1: y = y.mean(axis=1)

                        audio_arrays.append(y)
                        sampling_rates.append(sr)

            except Exception as e:
                print(f"Error: {e}")
                audio_arrays.append(np.array([0.0], dtype=np.float32))
                sampling_rates.append(16000)

        batch["audio"] = [{"array": arr, "sampling_rate": sr} for arr, sr in zip(audio_arrays, sampling_rates)]
        return batch


class TalkerAudioLoader(RelationalAudioLoader):
    def __call__(self, batch):
        batch = super().__call__(batch)
        
        target_arrays = []
        target_srs = []
        speaker_embeddings = []
        
        for session_id, t_start, t_end in zip(batch['session_id'], batch['target_start_sec'], batch['target_end_sec']):
            try:
                row_idx = self.id_to_idx.get(session_id)
                
                clean_entry = self.audio_dataset[row_idx]["clean_audio"] 
                raw_bytes = clean_entry['bytes']

                with io.BytesIO(raw_bytes) as file_obj:
                    with sf.SoundFile(file_obj) as f:
                        sr = f.samplerate
                        start_frame = int(t_start * sr)
                        frames_to_read = int((t_end - t_start) * sr)

                        if frames_to_read <= 0:
                            target_arrays.append(np.array([0.0], dtype=np.float32))
                            target_srs.append(sr)
                            speaker_embeddings.append(self.audio_dataset[row_idx]["speaker_embedding"])
                            continue

                        f.seek(start_frame)
                        y = f.read(frames=frames_to_read, dtype='float32')
                        if y.ndim > 1: y = y.mean(axis=1)

                        target_arrays.append(y)
                        target_srs.append(sr)
                
                # Load pre-computed speaker embedding for this session
                speaker_embeddings.append(self.audio_dataset[row_idx]["speaker_embedding"])

            except Exception as e:
                print(f"Target Audio Error: {e}")
                target_arrays.append(np.array([0.0], dtype=np.float32))
                target_srs.append(16000)
                # Re-raise since we don't have a valid speaker embedding fallback
                raise e
        
        batch["target_audio"] = [{"array": arr, "sampling_rate": sr} for arr, sr in zip(target_arrays, target_srs)]
        batch["speaker_embedding"] = speaker_embeddings
        return batch
