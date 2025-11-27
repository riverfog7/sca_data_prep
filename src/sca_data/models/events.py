from typing import Literal, Optional, Union, List

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")

    @property
    def duration(self) -> float:
        return self.end - self.start


class ComedianEvent(BaseEvent):
    role: Literal['comedian'] = 'comedian'
    content: str = Field(..., description="Verbatim transcript of speech OR description of the sound")
    event_type: Literal[
        'speech',
        'vocal_sfx',
        'laugh_self',
        'breath',
        'physical_sound',
        'singing',
        'other',
    ] = Field(..., description="Classification of the comedian's action")
    delivery_tag: Optional[str] = Field(None, description="Style of delivery (e.g., 'yelling', 'whispering', 'deadpan')")


class AudienceEvent(BaseEvent):
    role: Literal['audience'] = 'audience'
    content: str = Field(..., description="Transcript of speech OR description of noise")
    reaction_type: Literal[
        'laughter',
        'applause',
        'cheer',
        'heckle',
        'crowd_answer',
        'booing',
        'groan',
        'gasp',
        'other',
    ] = Field(..., description="Classification of the interaction")


class EnvironmentEvent(BaseEvent):
    role: Literal['environment'] = 'environment'
    content: str = Field(..., description="Description of the sound (e.g., 'Glass breaking', 'Feedback squeal')")
    sound_category: Literal[
        'music',
        'accidental_noise',
        'technical_noise',
        'ambient_sound',
        'other',
    ] = Field(..., description="Category of the environmental sound")


class ComedySession(BaseModel):
    video_id: str
    timeline: List[Union[ComedianEvent, AudienceEvent, EnvironmentEvent]]
