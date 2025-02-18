import librosa
import numpy as np
import pickle
from sklearn.svm import SVC

def extract_features(file_path):
    """ Extract MFCC features from voice sample """
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Prepare training data
real_samples = ["dataset/voice_samples/user1.wav", "dataset/voice_samples/user2.wav"]
fake_samples = ["dataset/replay_samples/replay1.wav", "dataset/replay_samples/replay2.wav"]

X = [extract_features(f) for f in real_samples + fake_samples]
y = [1] * len(real_samples) + [0] * len(fake_samples)  # 1=Real, 0=Fake

# Train SVM model
clf = SVC(kernel="linear", probability=True)
clf.fit(X, y)

# Save model
with open("models/voice_anti_spoof.pkl", "wb") as f:
    pickle.dump(clf, f)
