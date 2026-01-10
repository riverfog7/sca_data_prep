from sca_data.dataset_utils import preprocess_dataset_to_24k
from pathlib import Path

DATA_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")
preprocess_dataset_to_24k(DATA_DIR)