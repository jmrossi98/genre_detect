import os
import math
import json
import librosa

from settings import (
    SAMPLE_RATE,
    NUM_MFCC,
    N_FTT,
    HOP_LENGTH,
    NUM_SEGMENTS,
    DURATION,
)

DATASET_PATH = "data\\archive\\Data\\genres_original" # loaded using the GTZAN Music Genre Classification dataset at https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
JSON_PATH = "data\\data.json"

SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def dump_mfccs_to_json():
    """
    Processes test data as MFCCs and labels
    """
    data = {
        "mapping": [],
        "mfcc": [],
        "labels" : [],
    }
    samples_per_segment = int(SAMPLES_PER_TRACK/NUM_SEGMENTS)
    expected_mfcc = math.ceil(samples_per_segment/HOP_LENGTH)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):
        if dirpath is not DATASET_PATH:
            dirpath_components = dirpath.split("\\")
            label = dirpath_components[-1]
            data["mapping"].append(label)
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(NUM_SEGMENTS):
                    start_sample = samples_per_segment * s
                    finish_sample = start_sample + samples_per_segment
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=sr, n_fft=N_FTT, n_mfcc=NUM_MFCC, hop_length=HOP_LENGTH)
                    mfcc = mfcc.T
                    if len(mfcc) == expected_mfcc:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    dump_mfccs_to_json()