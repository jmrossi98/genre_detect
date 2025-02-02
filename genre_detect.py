import os
import sys
import keras
import librosa
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

    def __init__(self, model_name):
        self.model_name = model_name if model_name is not None else MODEL_NAME

    def train_and_save_model(self):
        test_splits: TestSplits = prepare_datasets(TEST_SIZE, VALIDATION_SIZE)
        input_shape = (test_splits.x_train.shape[1],test_splits.x_train.shape[2])
        model = build_model(input_shape)
        model = train_model(model, test_splits, model_name=self.model_name, plot_history=True)
        return model

    def process_input(self, audio_file, duration, segment):
        SAMPLES_PER_TRACK = SAMPLE_RATE * duration
        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
        start = samples_per_segment * segment
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T
        return mfcc

    def predict_with_new_sample(self, file_path):
        self.model = keras.models.load_model(f"models\\{self.model_name}.h5")

        # Predict all segments
        predictions = []
        for s in range(NUM_SEGMENTS):
            print(f"Processing segment {s+1}/{NUM_SEGMENTS}...")
            input_mfcc = self.process_input(file_path, DURATION, s)
            input_mfcc = input_mfcc[np.newaxis, ...]
            prediction = self.model.predict(input_mfcc)
            predicted_index = np.argmax(prediction, axis=1)
            predictions.append(int(predicted_index))

        # Get most common prediction indices
        predicted_index_overall = max(set(predictions), key=predictions.count)
        predicted_genre = GENRES[int(predicted_index_overall)]
        print("\nPredicted Genre:", predicted_genre)
        return predicted_genre


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
        '--file_path',
        help="Path to file you'd like to classify"
    )
    parser.add_argument(
        '--model_path',
        action="store_true",
        help="Path to custom model, uses my own rnn_genre_classifier by default"
    )
    parser.add_argument(
        '--build',
        action="store_true",
        help="Specify whether to build model"
    )
    parser.add_argument(
        '--preprocess',
        default=None,
        help="Path to raw data to preprocess for model to use as datasets (looks for unzipped GTZAN dataset in data folder by default)",
        metavar='PATH'
    )
    args = parser.parse_args()
    classifier = GenreClassifier(args.model_path)

    if len(sys.argv) == 1:
        parser.print_help()

    if args.preprocess:
        dump_mfccs_to_json(args.preprocess)

    if args.build is True:
        classifier.train_and_save_model()

    if args.file_path is not None:
        file_path = args.file_path
        if not os.path.exists(file_path):
            print(f"File at {file_path} does not exist")
        if not os.path.isfile(file_path):
            print(f'{file_path} is not a file')
        classifier.predict_with_new_sample(file_path)
        