import os
import math
import json
import librosa

DATASET_PATH = "data\\archive\\Data\\genres_original" # loaded using the GTZAN Music Genre Classification dataset at https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
JSON_PATH = "data\\data.json"

DURATION = 30
SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def dump_mfccs_to_json(num_segments=10, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Processes test data as MFCCs and labels
    """
    data = {
        "mapping": [],
        "mfcc": [],
        "labels" : [],
    }
    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    expected_mfcc = math.ceil(num_samples_per_segment/hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):
        if dirpath is not DATASET_PATH:
            dirpath_components = dirpath.split("\\")
            label = dirpath_components[-1]
            data["mapping"].append(label)
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                    mfcc = mfcc.T
                    if len(mfcc) == expected_mfcc:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    dump_mfccs_to_json()