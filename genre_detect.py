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
NEW_SAMPLES_PATH = "C:\\Users\\JakeR\\Downloads\\predict_data"

class GenreClassifier:

    def __init__(self, model_name=None):
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

    def test_model_new_samples(self):
        content = ""
        label_averages = []
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(NEW_SAMPLES_PATH)):
            if dirpath is not NEW_SAMPLES_PATH:
                dirpath_components = dirpath.split("\\")
                label = dirpath_components[-1]
                label_pass = 0
                label_samples = 0
                print(f"Testing {label} genre...")
                for f in filenames:
                    print(f"Testing {f}...")
                    file_path = os.path.join(dirpath, f)
                    pred_label = self.predict_with_new_sample(file_path)
                    print(f"Expected: {label}, Predicted: {pred_label}")
                    if label == pred_label:
                        label_pass += 1
                    label_samples +=1
                label_average = label_pass / label_samples
                print(f"Accuracy for {label}: {label_average}")
                content += f"{label}: {label_average}\n"
                label_averages.append(label_average)
        overall_avg = sum(label_averages) / len(label_averages)
        print(f"Overall Accuracy: {overall_avg}")
        content += f"\nOverall: {overall_avg}\n"
        with open("test_report.txt", "w") as f:
            f.write(content)


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
        default=None,
        help="Path to raw data to preprocess for model to use as datasets (looks for unzipped GTZAN dataset in data folder by default)",
        metavar='PATH'
    )
    args = parser.parse_args()
    classifier = GenreClassifier(args.name)

    classifier.test_model_new_samples()

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    # if args.preprocess:
    #     dump_mfccs_to_json(args.preprocess)

    # if args.build is True:
    #     classifier.train_and_save_model()

    # if args.path is not None:
    #     path = args.path
    #     if not os.path.exists(path):
    #         print(f"File at {path} does not exist")
    #         sys.exit(1)
    #     if not os.path.isfile(path):
    #         print(f'{path} is not a file')
    #         sys.exit(1)
    #     classifier.predict_with_new_sample(path)
    # sys.exit(0)
        