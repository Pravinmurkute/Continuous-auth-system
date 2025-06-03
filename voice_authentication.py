import torch
import torchaudio

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def authenticate_voice(model, audio_path, device):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)
    output = model(waveform)
    return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "voice_model.pth"
    audio_path = "test.wav"

    model = load_model(model_path, device)
    result = authenticate_voice(model, audio_path, device)
    print("Voice Authentication Output:", result)
