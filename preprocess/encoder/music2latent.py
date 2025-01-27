import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from music2latent import EncoderDecoder  # Import your custom model

class FeatureExtractorM2L:
    def __init__(self, device_id=0, sr=44100):
        self.device_id = device_id
        self.sr = sr
        self.device = torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")
        self.model = EncoderDecoder(device=self.device)

    def extract_features_from_segment(self, segment, sample_rate, save_path):
        input_audio = segment.unsqueeze(0).to(self.device)  # Add batch dimension and move to the device

        with torch.no_grad():
            model_outputs = self.model.encode(input_audio, extract_features=True)

        features = model_outputs.mean(dim=-1).cpu().numpy()
        np.save(save_path, features)


            
            
        
        
        