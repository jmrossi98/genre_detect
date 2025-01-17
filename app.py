import os
from flask import Flask, request, jsonify
import numpy as np
import keras
from keras.models import load_model
import librosa

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

app = Flask(__name__)
model_path = os.path.join(os.getcwd(), "models", "rnn_genre_classifier.h5")
model = load_model(model_path)

# Process segment
def process_audio(file, duration, segment):
    SAMPLES_PER_TRACK = SAMPLE_RATE * duration
    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    signal, sample_rate = librosa.load(file, sr=SAMPLE_RATE)
    start = samples_per_segment * segment
    finish = start + samples_per_segment
    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T
    return mfcc

# Process file segments
def predict_genre(file):
    # Predict all segments
    predictions = []
    for s in range(NUM_SEGMENTS):
        print(f"Processing segment {s+1}/{NUM_SEGMENTS}...")
        input_mfcc = process_audio(file, DURATION, s)
        input_mfcc = input_mfcc[np.newaxis, ...]
        prediction = model.predict(input_mfcc)
        predicted_index = np.argmax(prediction, axis=1)
        predictions.append(int(predicted_index))

    # Get most common prediction indices
    predicted_index_overall = max(set(predictions), key=predictions.count)
    predicted_genre = GENRES[int(predicted_index_overall)]
    return predicted_genre

# Define an endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Process the audio file and make a prediction
        genre = predict_genre(file)
        return jsonify({'predicted_genre': genre})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000)
    print(predict_genre("C:\\Users\\JakeR\\Downloads\\ny_state_of_mind.mp3"))