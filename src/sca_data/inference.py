from pathlib import Path
from typing import List

import librosa
import numpy as np
import tensorflow_hub as hub
import whisper

from .constants import YAMNET_TO_REACTION, FRAME_HOP_SEC, WINDOW_SEC, DEVICE, WHISPER_MODEL, WHISPER_CACHE, YAMNET_HUB_URL
from .models.events import ComedianEvent, AudienceEvent, ComedySession
from .utils import class_names_from_csv, to_comedian_event, build_comedy_session

whisper_model = whisper.load_model(WHISPER_MODEL, download_root=WHISPER_CACHE, device=DEVICE)

yamnet_model = hub.load(YAMNET_HUB_URL)
class_map_path = yamnet_model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)


def infer_comedian_events(file_path: Path) -> List[ComedianEvent]:
    result = whisper_model.transcribe(file_path.absolute().as_posix())

    events = []
    for segment in result["segments"]:
        events.append(to_comedian_event(segment))

    return events


def run_yamnet(audio_path: str) -> np.ndarray:
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    waveform = waveform.astype(np.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()   # [num_frames, num_classes]
    return scores_np


def extract_audience_events(scores_np,
                            score_thresh: float = 0.3,
                            min_duration: float = 0.3) -> List[AudienceEvent]:
    """
    YAMNet scores → AudienceEvent 리스트로 변환.

    - 각 프레임마다 YAMNET_TO_REACTION에 정의된 클래스 중
      가장 score가 높은 클래스를 고름
    - 그 score가 score_thresh 이상이면 그 프레임은 해당 reaction_type으로 active
    - 같은 reaction_type이 연속된 프레임들을 하나의 이벤트로 묶음
    - 이벤트 길이가 min_duration 미만이면 버림
    - JSON 스키마에 맞게:
      { start, end, role: "audience", content, reaction_type } 형태 반환
    """

    num_frames, num_classes = scores_np.shape

    # 1) YAMNet class index → reaction_type 매핑 (없으면 None)
    name_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_reaction = [None] * num_classes

    for yamnet_name, reaction in YAMNET_TO_REACTION.items():
        idx = name_to_index.get(yamnet_name)
        if idx is not None:
            index_to_reaction[idx] = reaction

    # 2) 프레임별로 "이 프레임에서 유효한 reaction_type" 결정
    #    (가장 score 높은 mapped 클래스 하나만)
    active_reaction_per_frame = [None] * num_frames

    for i in range(num_frames):
        frame_scores = scores_np[i]

        best_score = 0.0
        best_reaction = None

        # reaction이 정의된 클래스들만 확인
        for class_idx, reaction in enumerate(index_to_reaction):
            if reaction is None:
                continue
            s = frame_scores[class_idx]
            if s > best_score:
                best_score = s
                best_reaction = reaction

        if best_reaction is not None and best_score >= score_thresh:
            active_reaction_per_frame[i] = best_reaction
        # 아니면 None 그대로 (이 프레임은 audience reaction 없음)

    # 3) 연속된 프레임들을 같은 reaction_type으로 묶어서 이벤트 생성
    events = []
    current_reaction = None
    start_frame = None

    for i, reaction in enumerate(active_reaction_per_frame):
        if reaction != current_reaction:
            # 이전 이벤트가 끝난 경우 마감
            if current_reaction is not None and start_frame is not None:
                end_frame = i - 1

                start_t = start_frame * FRAME_HOP_SEC
                end_t = end_frame * FRAME_HOP_SEC + WINDOW_SEC
                duration = end_t - start_t

                if duration >= min_duration:
                    events.append(AudienceEvent(
                        start=float(start_t),
                        end=float(end_t),
                        content=f"[{current_reaction}]",
                        reaction_type=current_reaction,
                    ))

            # 새 이벤트 시작 (reaction가 None이면 그냥 쉬는 구간)
            current_reaction = reaction
            start_frame = i if reaction is not None else None

    # 4) 마지막 프레임까지 active였던 이벤트 처리
    if current_reaction is not None and start_frame is not None:
        end_frame = num_frames - 1
        start_t = start_frame * FRAME_HOP_SEC
        end_t = end_frame * FRAME_HOP_SEC + WINDOW_SEC
        duration = end_t - start_t

        if duration >= min_duration:
            events.append(AudienceEvent(
                start=float(start_t),
                end=float(end_t),
                content=f"[{current_reaction}]",
                reaction_type=current_reaction,
            ))

    return events


def infer_audience_events(file_path: Path,
                          score_thresh: float = 0.3,
                          min_duration: float = 0.3) -> List[AudienceEvent]:
    scores_np = run_yamnet(file_path.absolute().as_posix())
    audience_events = extract_audience_events(
        scores_np,
        score_thresh=score_thresh,
        min_duration=min_duration,
    )
    return audience_events


def infer_comedy_session(file_path: Path) -> ComedySession:
    comedian_events = infer_comedian_events(file_path)
    audience_events = infer_audience_events(file_path)

    video_id = file_path.stem

    comedy_session = build_comedy_session(
        video_id=video_id,
        comedian_events=comedian_events,
        audience_events=audience_events,
    )

    return comedy_session
