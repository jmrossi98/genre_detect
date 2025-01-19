import os
from flask import Flask, request, jsonify
import numpy as np
import keras
from keras.models import load_model
import librosa
import traceback

from settings import (
    GENRES,
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

# App and model
app = Flask(__name__)
model_path = os.path.join(os.getcwd(), "models", "rnn_genre_classifier.h5")
model = load_model(model_path)

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Process a single segment of audio
def process_audio_segment(signal, sample_rate, duration, segment):
    """
    Processes a specific segment of the loaded audio signal and computes MFCCs.

    Args:
        signal (np.ndarray): The loaded audio signal.
        sample_rate (int): The sample rate of the audio.
        duration (int): The duration of the audio in seconds.
        segment (int): The segment index to process.

    Returns:
        np.ndarray: The MFCCs for the segment.
    """
    SAMPLES_PER_TRACK = SAMPLE_RATE * duration
    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    start = samples_per_segment * segment
    finish = start + samples_per_segment
    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
    return mfcc.T

# Process file and predict genre
def predict_genre(file):
    """
    Predicts the genre of an audio file using the model.

    Args:
        file (str): Path to the audio file.

    Returns:
        str: Predicted genre.
    """
    # Load the audio file once
    signal, sample_rate = librosa.load(file, sr=SAMPLE_RATE, duration=DURATION)

    # Predict all segments
    predictions = []
    for s in range(NUM_SEGMENTS):
        print(f"Processing segment {s+1}/{NUM_SEGMENTS}...")
        input_mfcc = process_audio_segment(signal, sample_rate, DURATION, s)
        input_mfcc = input_mfcc[np.newaxis, ...]
        prediction = model.predict(input_mfcc)
        predicted_index = np.argmax(prediction, axis=1)
        predictions.append(int(predicted_index))

    # Get most common prediction indices
    predicted_index_overall = max(set(predictions), key=predictions.count)
    predicted_genre = GENRES[int(predicted_index_overall)]
    return predicted_genre


# Define an endpoint to make predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Check if the file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        # Check if the file is selected
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process the file
        predicted_genre = predict_genre(filepath)

        # Clean up the temporary file
        os.remove(filepath)

        return jsonify({"genre": predicted_genre})
    except Exception as e:
        return jsonify({"error": traceback.format_exc()}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)