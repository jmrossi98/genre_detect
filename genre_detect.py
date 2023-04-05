import os
import sys
import keras
import librosa
import argparse
import numpy as np

from settings import (
    TEST_SIZE,
    VALIDATION_SIZE,
    SAMPLE_RATE,
    NUM_MFCC,
    N_FTT,
    HOP_LENGTH,
    NUM_SEGMENTS,
    DURATION,
    TestSplits,
)

from src.preprocess_data import dump_mfccs_to_json
from src.prepare_dataset import prepare_datasets
from src.build_model import build_model
from src.train_model import train_model

MODEL_NAME = "rnn_genre_classifier"
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

class GenreClassifier:

    def __init__(self, model_name=None):
        self.model_name = model_name if model_name is not None else MODEL_NAME

    def train_and_save_model(self):
        test_splits: TestSplits = prepare_datasets(TEST_SIZE, VALIDATION_SIZE)
        input_shape = (test_splits.x_train.shape[1],test_splits.x_train.shape[2])
        model = build_model(input_shape)
        model = train_model(model, test_splits, model_name=self.model_name, plot_history=True)
        return model

    def process_input(self, audio_file, duration):
        SAMPLES_PER_TRACK = SAMPLE_RATE * duration
        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
        for s in range(NUM_SEGMENTS):
            start = samples_per_segment * s
            finish = start + samples_per_segment
            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
            mfcc = mfcc.T
            return mfcc

    def predict_with_new_sample(self, file_path):
        self.model = keras.models.load_model(f"models\\{self.model_name}")
        input_mfcc = self.process_input(file_path, DURATION)
        input_mfcc = input_mfcc[np.newaxis, ...]
        prediction = self.model.predict(input_mfcc)
        predicted_index = np.argmax(prediction, axis=1)
        print("Predicted Genre:", GENRES[int(predicted_index)])


if __name__ == "__main__":
    classifier = GenreClassifier()
    print("""
    ,----.                                ,------.         ,--.              ,--.   
    '  .-./   ,---.,--,--,,--.--.,---.     |  .-.  \ ,---.,-'  '-.,---. ,---,-'  '-. 
    |  | .---| .-. |      |  .--| .-. :    |  |  \  | .-. '-.  .-| .-. | .--'-.  .-' 
    '  '--'  \   --|  ||  |  |  \   --.    |  '--'  \   --. |  | \   --\ `--. |  |   
    `------' `----`--''--`--'   `----'    `-------' `----' `--'  `----'`---' `--' 
    """)
    parser = argparse.ArgumentParser(description="Give a path to a music file and we'll predict the genre", usage=argparse.SUPPRESS)
    parser.add_argument(
        '--path',
        help="Path to file you'd like to classify"
    )
    parser.add_argument(
        '--name',
        help="Name of model in models folder (only if you built custom model, uses rnn_genre_classifier by default)"
    )
    parser.add_argument(
        '--build',
        action="store_true",
        help="Specify whether to build model"
    )
    parser.add_argument(
        '--preprocess',
        help="Path to raw data to preprocess for model to use as datasets (looks for unzipped GTZAN dataset in data folder by default)",
        metavar='PATH'
    )
    args = parser.parse_args()
    classifier = GenreClassifier(args.name)

    if args.preprocess:
        dump_mfccs_to_json(args.preprocess)

    if args.build is True:
        classifier.train_and_save_model()

    if args.path is not None:
        path = args.path
        if not os.path.exists(path):
            print(f"File at {path} does not exist")
            sys.exit(1)
        if not os.path.isfile(path):
            print(f'"{path}" is not a file')
            sys.exit(1)
        classifier.predict_with_new_sample(path)
        sys.exit(0)
    else:
        parser.print_help()
        