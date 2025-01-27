import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel

class FeatureExtractorMERT:
    def __init__(self, model_name="m-a-p/MERT-v1-95M", device_id=0, sr=24000):
        self.model_name = model_name
        self.device_id = device_id
        self.sr = sr
        self.device = torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name, trust_remote_code=True)

    def extract_features_from_segment(self, segment, sample_rate, save_path):
        input_audio = segment.float()
        model_inputs = self.processor(input_audio, sampling_rate=sample_rate, return_tensors="pt")
        model_inputs = model_inputs.to(self.device)

        with torch.no_grad():
            model_outputs = self.model(**model_inputs, output_hidden_states=True)

        # Stack and process hidden states
        all_layer_hidden_states = torch.stack(model_outputs.hidden_states).squeeze()[1:, :, :].unsqueeze(0)
        all_layer_hidden_states = all_layer_hidden_states.mean(dim=2)
        features = all_layer_hidden_states.cpu().detach().numpy()

        # Save features
        np.save(save_path, features)