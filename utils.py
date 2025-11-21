import base64
import io
import re
from pathlib import Path
from typing import Optional

import soundfile as sf

from models.audio import AudioSlice


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
