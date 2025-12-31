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
from datasets import Features, Value, Audio
from tqdm import tqdm

from .constants import DEFAULT_SYSTEM_PROMPT, DEFAULT_INSTRUCTION_PROMPT
from .models.events import ComedianEvent, BaseEvent, AudienceEvent, EnvironmentEvent, ComedySession
from .utils import clean_audio_bytes, check_and_resample_audio


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


def to_hf_dataset(sessions: Iterable[ComedySession], audio_base_path: Path, min_duration: float, max_duration: float) -> DatasetDict:
    event_rows = []
    unique_sessions = {}
    for session in sessions:
        if session.video_id not in unique_sessions:
            audio_path = list(audio_base_path.glob(f"{session.video_id}.*"))
            if len(audio_path) != 1:
                raise FileNotFoundError(f"Audio file not found for session {session.video_id} in {audio_base_path}")
            unique_sessions[session.video_id] = str(audio_path[0])

        for i, event in enumerate(session.timeline):
            if isinstance(event, ComedianEvent) and event.event_type == 'speech':
                # Don't include very short or very long segments
                if event.start < min_duration or event.start > max_duration:
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
            with open(path, "rb") as f:
                original_bytes = f.read()
            
            try:
                # Moshi's mimi neural audio codec takes 24kHz input
                cleaned_bytes = clean_audio_bytes(original_bytes, target_sr=24000)
                print(f"Cleaned audio for {sess_id} successfully.")
            except Exception as e:
                print(f"Warning: Failed to clean audio for {sess_id}, using original. Error: {e}")
                cleaned_bytes = original_bytes 
            yield {
                "session_id": sess_id,
                "audio": {"path": path, "bytes": check_and_resample_audio(original_bytes, target_sr=16000)},    # 원본 (Context용)
                "clean_audio": {"path": None, "bytes": cleaned_bytes} # AI 처리됨 (Target용)
            }

    
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
        "clean_audio": Audio(decode=False)  # Clean 버전 추가
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
    
    for input_audio, target_audio, text in zip(batch["audio"], batch["target_audio"], batch["target_text"]):
        msgs = [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_waveform": input_audio["array"], "sampling_rate": 16000},
                    {"type": "text", "text": instruction_prompt or DEFAULT_INSTRUCTION_PROMPT},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text.strip()},
                    {"type": "audio", "audio_waveform": target_audio["array"], "sampling_rate": 24000}
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
                    "sampling_rate": 16000
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
                            continue

                        f.seek(start_frame)
                        y = f.read(frames=frames_to_read, dtype='float32')
                        if y.ndim > 1: y = y.mean(axis=1)

                        target_arrays.append(y)
                        target_srs.append(sr)

            except Exception as e:
                print(f"Target Audio Error: {e}")
                target_arrays.append(np.array([0.0], dtype=np.float32))
                target_srs.append(16000)
        
        batch["target_audio"] = [{"array": arr, "sampling_rate": sr} for arr, sr in zip(target_arrays, target_srs)]
        return batch
