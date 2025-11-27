import io

import numpy as np
import soundfile as sf


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
