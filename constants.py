import torch
from pathlib import Path

PROCESS_EXTS = [".flac", ".wav", ".mp3", ".m4a", ".webm"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WHISPER_MODEL = "large-v3-turbo"
WHISPER_CACHE = (Path(__file__).parent.resolve() / "./.cache/whisper").absolute().as_posix()

YAMNET_HUB_URL = 'https://tfhub.dev/google/yamnet/1'

FRAME_HOP_SEC = 0.48   # YAMNet hop size
WINDOW_SEC    = 0.96   # YAMNet window size

YAMNET_TO_REACTION = {
    # 웃음 계열
    "Laughter": "laughter",
    "Baby laughter": "laughter",
    "Giggle": "laughter",
    "Snicker": "laughter",
    "Belly laugh": "laughter",
    "Chuckle, chortle": "laughter",

    # 박수 계열
    "Applause": "applause",
    "Clapping": "applause",

    # 환호 / 응원
    "Cheering": "cheer",
    "Chant": "cheer",

    # 탄식 / 신음
    "Groan": "groan",
    "Wail, moan": "groan",
    "Sigh": "groan",
    "Whimper": "groan",

    # 놀람
    "Gasp": "gasp",

    # 군중 발화 (대답/웅성거림 계열)
    "Child speech, kid speaking": "crowd_answer",
    "Conversation": "crowd_answer",
    "Crowd": "crowd_answer",
    "Hubbub, speech noise, speech babble": "crowd_answer",
    "Children shouting": "crowd_answer",

    # 공격적인 외침 (heckle 후보)
    "Shout": "heckle",
    "Yell": "heckle",
    "Bellow": "heckle",
    "Screaming": "heckle",
}
