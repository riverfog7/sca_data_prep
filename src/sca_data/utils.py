import base64
import csv
import io
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
import torchaudio
from clearvoice import ClearVoice

from .models.audio import AudioSlice
from .models.events import ComedianEvent, ComedySession, AudienceEvent

# 전역 변수: 모델을 매번 로드하지 않고 캐싱하기 위함
_CLEARVOICE_MODEL = None
def get_clearvoice_model():
    global _CLEARVOICE_MODEL
    if _CLEARVOICE_MODEL is None:
        print("[Info] Loading ClearerVoice-Studio model (MossFormer2)...")
        _CLEARVOICE_MODEL = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
    return _CLEARVOICE_MODEL

def clean_audio_bytes(raw_bytes: bytes, target_sr: int = 24000) -> bytes:
    """
    Remove noise using ClearerVoice-Studio (MossFormer2).
    Logic: Load -> Resample(48k) -> Slice(60s) -> Process -> Concat -> Resample(16k) -> Bytes
    """
    cv_model = get_clearvoice_model()

    with io.BytesIO(raw_bytes) as bio:
        wav, sr = torchaudio.load(bio)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.shape[0] == 1:
        pass
    else:
        wav = wav.unsqueeze(0)

    # ClearVoice model requires 48kHz
    target_model_sr = 48000
    if sr != target_model_sr:
        resampler = torchaudio.transforms.Resample(sr, target_model_sr)
        wav_48k = resampler(wav)
    else:
        wav_48k = wav

    #Slicing by Chunk
    CHUNK_SECONDS = 15  # 15 seconds
    CHUNK_SIZE = target_model_sr * CHUNK_SECONDS  # 48000 * 15 = 720,000 samples
    
    total_samples = wav_48k.shape[1]
    input_numpy_full = wav_48k.numpy() # [1, Total_Time]
    
    enhanced_chunks = []
    
    for start_idx in range(0, total_samples, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, total_samples)
        
        chunk = input_numpy_full[:, start_idx:end_idx]
        
        try:
            enhanced_out = cv_model(input_path=chunk, online_write=False)
            
            if isinstance(enhanced_out, dict):
                res = list(enhanced_out.values())[0]
            elif isinstance(enhanced_out, list):
                res = enhanced_out[0]
            else:
                res = enhanced_out
            if res.ndim == 1:
                res = res[None, :]
            
            enhanced_chunks.append(res)
            
        except Exception as e:
            print(f"   Warning: Chunk failed ({start_idx}~{end_idx}). Using original. Error: {e}")
            enhanced_chunks.append(chunk) # if it fail to process, use original chunk
        
        torch.cuda.empty_cache()
       
    #Combine Chunks
    if len(enhanced_chunks) > 0:
        enhanced_numpy = np.concatenate(enhanced_chunks, axis=1)
    else:
        enhanced_numpy = input_numpy_full

    # Numpy -> Tensor 변환
    enhanced_tensor = torch.from_numpy(enhanced_numpy)
    
    # 16kHz Downsampling (원복)
    if target_model_sr != target_sr:
        resampler_back = torchaudio.transforms.Resample(target_model_sr, target_sr)
        final_tensor = resampler_back(enhanced_tensor)
    else:
        final_tensor = enhanced_tensor
    
    final_numpy_1d = final_tensor.squeeze().numpy()
    
    with io.BytesIO() as out_bio:
        sf.write(out_bio, final_numpy_1d, target_sr, format="FLAC")
        clean_bytes = out_bio.getvalue()
        
    return clean_bytes

def check_and_resample_audio(raw_bytes: bytes, target_sr: int = 16000) -> bytes:
    with io.BytesIO(raw_bytes) as bio:
        wav, sr = torchaudio.load(bio)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.shape[0] == 1:
        pass
    else:
        wav = wav.unsqueeze(0)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav_resampled = resampler(wav)
    else:
        wav_resampled = wav

    with io.BytesIO() as out_bio:
        sf.write(out_bio, wav_resampled.squeeze().numpy(), target_sr, format="FLAC")
        resampled_bytes = out_bio.getvalue()
    return resampled_bytes

def get_video_id(url: str) -> Optional[str]:
    matched = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:&|\/|$)", url)
    if matched:
        return matched.group(1)
    return None

def get_sliced_audio_base64(base_path: Path, slice: AudioSlice) -> str:
    audio_path = base_path / slice.file
    info = sf.info(audio_path)
    sr = info.samplerate

    start_frame = int(slice.start_time * sr)
    stop_frame = int(slice.end_time * sr)

    data, _ = sf.read(audio_path, start=start_frame, stop=stop_frame, dtype='int16')

    buffered = io.BytesIO()
    sf.write(buffered, data, sr, format='WAV')

    return base64.b64encode(buffered.getvalue()).decode()


def cut_audio_base64(base_path: Path, slice: AudioSlice, sample_interval: int = 4) -> List[str]:
    """
    Cut audio into segments optimized for Qwen3-Omni AuT encoder.

    Args:
        sample_interval: Duration of each segment in seconds (default: 6s)
                        Recommended range: 4-8 seconds for AuT's attention window
    """
    audio_path = base_path / slice.file
    info = sf.info(audio_path)
    sr = info.samplerate

    start_frame = int(slice.start_time * sr)
    stop_frame = int(slice.end_time * sr)

    total_frames = stop_frame - start_frame
    segment_frames = int(sample_interval * sr)

    segments = []
    for i in range(0, total_frames, segment_frames):
        segment_start = start_frame + i
        segment_stop = min(segment_start + segment_frames, stop_frame)

        data, _ = sf.read(audio_path, start=segment_start, stop=segment_stop, dtype='int16')

        buffered = io.BytesIO()
        sf.write(buffered, data, sr, format='WAV')
        segments.append(base64.b64encode(buffered.getvalue()).decode())

    return segments


def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names


def to_comedian_event(whisper_segment: dict) -> ComedianEvent:
    return ComedianEvent(
        event_type="speech",
        start=whisper_segment["start"],
        end=whisper_segment["end"],
        delivery_tag=None,
        content=whisper_segment['text'],
    )


def build_comedy_session(video_id: str, comedian_events: List[ComedianEvent], audience_events: List[AudienceEvent]) -> ComedySession:
    timeline = comedian_events + audience_events
    timeline.sort(key=lambda event: event.start)

    return ComedySession(
        video_id=video_id,
        timeline=timeline
    )
