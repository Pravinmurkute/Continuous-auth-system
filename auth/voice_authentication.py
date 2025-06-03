import numpy as np
import librosa
import scipy.spatial

class VoiceAuthenticator:
    def extract_voice_features(self, audio_file):
        """Extract MFCC voice features."""
        y, sr = librosa.load(audio_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0)
        return features

    def verify_voice(self, voice1, voice2, threshold=0.5):
        """Compare voice features using Euclidean distance."""
        distance = scipy.spatial.distance.euclidean(voice1, voice2)
        return distance < threshold
