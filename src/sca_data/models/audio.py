from pydantic import BaseModel


class AudioSlice(BaseModel):
    start_time: float
    end_time: float
    file: str

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class SlicedAudioFile(BaseModel):
    file: str
    slices: list[AudioSlice]

    @property
    def duration(self) -> float:
        return sum(slice.duration for slice in self.slices)
