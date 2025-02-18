import librosa
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load trained models
with open("models/voice_model.pkl", "rb") as f:
    voice_model = pickle.load(f)

with open("models/voice_anti_spoof.pkl", "rb") as f:
    anti_spoof_model = pickle.load(f)

def extract_features(audio_data, sr=16000):
    """ Extract MFCC features from voice sample """
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def verify_voice(audio_data):
    """ Verify voice authenticity and detect replay attacks """
    features = extract_features(audio_data)

    # Check if it's a real voice or a replay attack
    is_real = anti_spoof_model.predict([features])[0]
    if is_real == 0:
        print("Spoofing detected! Fake voice sample.")
        return False

    # Check if voice matches stored profile
    similarity = cosine_similarity([features], [voice_model["features"]])
    return similarity[0][0] > 0.7  # Threshold
