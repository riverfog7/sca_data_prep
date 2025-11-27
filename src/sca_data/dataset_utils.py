import io
from pathlib import Path
from typing import Iterable
from typing import Tuple

import numpy as np
import soundfile as sf
from datasets import DatasetDict, Dataset
from datasets import Features, Value, Audio

from .models.events import ComedianEvent, BaseEvent, AudienceEvent, EnvironmentEvent, ComedySession


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


def to_hf_dataset(sessions: Iterable[ComedySession], audio_base_path: Path) -> DatasetDict:
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
                event_rows.append({
                    "session_id": session.video_id,
                    "start_sec": 0.0,
                    "end_sec": event.start,
                    "target_text": event.content,
                    "event_index": i
                })
            else:
                raise ValueError(f"Unexpected event type in session {session.video_id}: {event}")


    def audio_generator():
        for sess_id, path in unique_sessions.items():
            with open(path, "rb") as f:
                yield {
                    "session_id": sess_id,
                    "audio": {"path": path, "bytes": f.read()}
                }

    text_features = Features({
        "session_id": Value("string"),
        "start_sec": Value("float"),
        "end_sec": Value("float"),
        "target_text": Value("string"),
        "event_index": Value("int32"),
    })
    audio_features = Features({
        "session_id": Value("string"),
        "audio": Audio(decode=False)
    })

    ds_text = Dataset.from_list(event_rows, features=text_features)
    ds_audio = Dataset.from_generator(audio_generator, features=audio_features)

    return DatasetDict({
        "storage": ds_audio,
        "train": ds_text,
    })


class RelationalAudioLoader:
    def __init__(self, audio_dataset):
        self.audio_lookup = {
            row['session_id']: row['audio']['bytes']
            for row in audio_dataset
        }

    def __call__(self, batch):
        audio_arrays = []
        sampling_rates = []

        for session_id, start, end in zip(batch['session_id'], batch['start_sec'], batch['end_sec']):
            try:
                raw_bytes = self.audio_lookup.get(session_id)

                if raw_bytes is None:
                    raise ValueError(f"Session {session_id} not found in storage")

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
