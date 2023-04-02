import json
import numpy as np
from sklearn.model_selection import train_test_split

DATASETS_PATH = "data\\data.json"

def prepare_datasets(test_size, validation_size):
    """
    Loads data into train, test and validation sets
    """

    with open(DATASETS_PATH, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test
