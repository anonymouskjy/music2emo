import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification
import numpy as np

class FeedforwardModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedforwardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),  
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
            nn.Dropout(0.3),  

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, output_size),
        )

    def forward(self, x):
        logit = self.model(x)
        return logit
    
