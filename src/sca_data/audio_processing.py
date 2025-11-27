from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from .models.audio import SlicedAudioFile, AudioSlice


def get_vad_slices(file_path: Path, slice_min: float = 5.0) -> SlicedAudioFile:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)

    (get_speech_timestamps, _, _, _, _) = utils

    try:
        data, sr = sf.read(file_path.absolute(), dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        target_sr = 16000
        if sr != target_sr:
            number_of_samples = round(len(data) * float(target_sr) / sr)
            data = scipy.signal.resample(data, number_of_samples)

        wav = torch.from_numpy(data)

    except Exception as e:
        print(f"Error loading audio with Soundfile/Scipy: {e}")
        return []

    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
    total_duration = len(wav) / 16000.0
    if not speech_timestamps:
        return [(0.0, total_duration)]

    cut_points = [0.0]

    for i in range(len(speech_timestamps) - 1):
        end_current = speech_timestamps[i]['end']
        start_next = speech_timestamps[i + 1]['start']
        midpoint = (end_current + start_next) / 2
        cut_points.append(midpoint)

    cut_points.append(total_duration)

    target_sec = slice_min * 60
    min_dur = target_sec * 0.8
    max_dur = target_sec * 1.2

    final_slices = []
    last_cut = 0.0
    i = 1
    while i < len(cut_points):
        candidate = cut_points[i]
        current_duration = candidate - last_cut

        if current_duration < min_dur:
            i += 1
            continue

        if i + 1 < len(cut_points):
            next_candidate = cut_points[i + 1]
            next_duration = next_candidate - last_cut
            curr_error = abs(current_duration - target_sec)
            next_error = abs(next_duration - target_sec)

            if (next_duration > max_dur) or (curr_error <= next_error):
                final_slices.append((last_cut, candidate))
                last_cut = candidate
        else:
            final_slices.append((last_cut, candidate))
            last_cut = candidate

        i += 1

    if last_cut < total_duration:
        if (total_duration - last_cut) > 0.1:
            if final_slices:
                last_start, _ = final_slices.pop()
                final_slices.append((last_start, total_duration))
            else:
                final_slices.append((0.0, total_duration))

    return SlicedAudioFile(file=file_path.absolute().name, slices=[AudioSlice(start_time=s[0], end_time=s[1], file=file_path.absolute().name) for s in final_slices])


