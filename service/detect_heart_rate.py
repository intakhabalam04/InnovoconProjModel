import os

import numpy as np
import librosa
import scipy.signal as sg
import matplotlib.pyplot as plt
from flask import jsonify
from tensorflow.keras.models import load_model


UPLOAD_FOLDER = 'uploaded_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_audio(audio_signal, sr, frame_duration=0.2, overlap=0.5):
    # Bandpass filter
    lowcut = 25.0
    highcut = 400.0
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sg.butter(1, [low, high], btype='band')
    filtered_signal = sg.filtfilt(b, a, audio_signal)

    # Normalize the signal
    normalized_signal = (filtered_signal - np.min(filtered_signal)) / (
                np.max(filtered_signal) - np.min(filtered_signal))

    frame_size_samples = int(frame_duration * sr)
    step_size = int((1 - overlap) * frame_size_samples)

    frames = []
    for start in range(0, len(normalized_signal) - frame_size_samples + 1, step_size):
        frame = normalized_signal[start:start + frame_size_samples]
        frames.append(frame)
    return np.array(frames), normalized_signal


def predict_peaks(file_path, model, sr=2000, frame_duration=0.2, overlap=0.5):
    # Load the audio signal
    audio_signal, _ = librosa.load(file_path, sr=sr, duration=100)

    # Preprocess the audio signal with overlap
    frames, normalized_signal = preprocess_audio(audio_signal, sr, frame_duration, overlap)

    # Reshape frames for model input (adding channel dimension)
    frames_reshaped = frames[..., np.newaxis]  # Add a channel dimension

    # Predict peaks using the model
    predicted_probs = model.predict(frames_reshaped)

    # Return raw probabilities and the normalized signal
    return predicted_probs.flatten(), normalized_signal


def group_high_prob_frames(predicted_probs, time_labels, threshold=95):
    high_prob_ranges = []
    start_idx = None

    for i, prob in enumerate(predicted_probs):
        if prob * 100 >= threshold:
            if start_idx is None:
                start_idx = i  # Start of a new high-probability region
        else:
            if start_idx is not None:
                # End of a high-probability region
                high_prob_ranges.append((time_labels[start_idx], time_labels[i] + 0.2))
                start_idx = None

    # If we end with a high-probability range, close it
    if start_idx is not None:
        high_prob_ranges.append((time_labels[start_idx], time_labels[len(predicted_probs) - 1] + 0.2))

    return high_prob_ranges


def plot_predictions(file_path, model, sr=2000, frame_duration=0.2, overlap=0.5):
    # Predict peaks for frames with 50% overlap
    predicted_probs, normalized_signal = predict_peaks(file_path, model, sr, frame_duration, overlap)

    # Create time arrays for the audio signal
    time_audio = np.linspace(0, len(normalized_signal) / sr, len(normalized_signal))

    # Create time labels for each frame
    frame_size_samples = int(frame_duration * sr)
    step_size = int((1 - overlap) * frame_size_samples)

    num_frames = len(predicted_probs)
    time_labels = [i * (step_size / sr) for i in range(num_frames)]  # Time at the start of each frame

    # Group frames with consecutive high probabilities (>= 95%) into larger frames
    high_prob_ranges = group_high_prob_frames(predicted_probs, time_labels, threshold=95)

    temp = []
    # Highlight only the max y value in each large frame
    for (frame_start, frame_end) in high_prob_ranges:
        # Get the portion of the audio signal in the current larger frame
        start_idx = np.searchsorted(time_audio, frame_start)
        end_idx = np.searchsorted(time_audio, frame_end)

        # Find the maximum y value in the current large frame
        if end_idx > start_idx:
            max_idx = np.argmax(normalized_signal[start_idx:end_idx]) + start_idx
            max_time = time_audio[max_idx]
            max_value = normalized_signal[max_idx]

            temp.append(max_time)
    # Count the number of detected peaks
    num_beats = len(temp)

    # Calculate duration of the analyzed segment (in seconds)
    total_duration = len(normalized_signal) / sr

    # Calculate heart rate (beats per minute)
    heart_rate_bpm = ((num_beats / total_duration) * 60) / 2

    print(f"Estimated Heart Rate: {heart_rate_bpm:.2f} BPM")
    return heart_rate_bpm

def find_heart_rate(request):

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            print(f"File saved to {file_path}")
            model_path = os.path.join(os.path.dirname(__file__), "../models/pcg.keras")
            print(model_path)
            model = load_model(model_path)
            heart_rate = plot_predictions(file_path, model, sr=2000, frame_duration=0.2, overlap=0.75)

            # Delete the files after sending the response
            os.remove(file_path)

            return jsonify({"heart_rate": round(heart_rate)}), 200
        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({"error": "Error processing file"}), 500

