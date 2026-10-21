import logging
from dataclasses import dataclass

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class DataArgs(Serializable):
    """
     Arguments for data loading. Train and eval data should be jsonl files
    with "path" and "duration" fields for each stereo audio .wav file.
    The target speaker and dialogue-context speaker are selected by channel index.
    """

    train_data: str = ""
    shuffle: bool = False
    eval_data: str = ""
    # Exclude whole source recordings at or above this duration before chunking.
    max_audio_duration_sec: float | None = None
    # Stereo input: generated/target speaker is channel 0, dialogue context is channel 1.
    target_audio_channel: int = 0
    context_audio_channel: int = 1

    def __post_init__(self) -> None:
        if self.max_audio_duration_sec is not None and self.max_audio_duration_sec <= 0:
            raise ValueError("max_audio_duration_sec must be greater than zero")
        if self.target_audio_channel < 0 or self.context_audio_channel < 0:
            raise ValueError("audio channel indices must be non-negative")
        if self.target_audio_channel == self.context_audio_channel:
            raise ValueError("target and context audio channels must be different")
