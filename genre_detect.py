import os
import sys
import keras
import librosa
import argparse
import numpy as np

from settings import (
    SAMPLE_RATE,
    NUM_MFCC,
    N_FTT,
    HOP_LENGTH,
    NUM_SEGMENTS,
    DURATION,
    TestSplits,
)

from src.prepare_dataset import prepare_datasets
from src.build_model import build_model
from src.train_model import train_model

MODEL_NAME = "rnn_genre_classifier.h5"
GENRES = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

def process_input(audio_file, duration):
    SAMPLES_PER_TRACK = SAMPLE_RATE * duration
    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
    
    for s in range(NUM_SEGMENTS):
        start = samples_per_segment * s
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T
        return mfcc

class GenreClassifier:

    def __init__(self, file_path, load_model=True):
        self.file_path = file_path
        self.test_splits: TestSplits = prepare_datasets(0.3, 0.2)
        if load_model is False:
            input_shape = (self.test_splits.x_train.shape[1], self.test_splits.x_train.shape[2])
            self.model = build_model(input_shape)
        else:
            self.model = keras.models.load_model(f"models\\{MODEL_NAME}")

    # Added as proof for how I trained my model
    def train_and_save_model(self):
        self.model = train_model(self.model, self.test_splits, model_name=MODEL_NAME, plot_history=True)

    def predict_with_new_sample(self):
        input_mfcc = process_input(self.file_path, DURATION)
        input_mfcc = input_mfcc[np.newaxis, ...]
        prediction = self.model.predict(input_mfcc)

        predicted_index = np.argmax(prediction, axis=1)
        print("Predicted Genre:", GENRES[int(predicted_index)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Give a path to an audio file...')
    parser.add_argument(
        '--path', '-p',
        help="Path to file you'd like to classify"
    )
    args = parser.parse_args()

    if args.path is not None:
        path = args.path

        if not os.path.exists(path):
            print(f"File at {path} does not exist")
            sys.exit(1)
        if not os.path.isfile(path):
            print(f'"{path}" is not a file')
            sys.exit(1)

        classifier = GenreClassifier(path)
        classifier.predict_with_new_sample()
        sys.exit(0)
        
