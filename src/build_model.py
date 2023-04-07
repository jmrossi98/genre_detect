from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Dense
from keras.optimizers import Adam

def build_model(input_shape):
    """
    Builds model
    """

    # build network
    model = Sequential()

    # add layers
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))

    # compile network
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.summary()

    return model
