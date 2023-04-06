import numpy as np
from dataclasses import dataclass

TEST_SIZE = 0.25
VALIDATION_SIZE = 0.20

SAMPLE_RATE = 22050
NUM_MFCC = 13
N_FTT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10
DURATION = 30
BATCH_SIZE = 32
EPOCHS = 200

@dataclass
class TestSplits:
    x_train: np.array
    x_validation: np.array
    x_test: np.array
    y_train: np.array
    y_validation: np.array
    y_test: np.array
