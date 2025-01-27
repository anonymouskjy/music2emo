import os
import numpy as np
import pickle
from torch.utils import data
import torchaudio.transforms as T
import torchaudio
import torch
import csv
import pytorch_lightning as pl
from music2latent import EncoderDecoder
import json
import math
from sklearn.preprocessing import StandardScaler
import pandas as pd

class PMEmoDataset(data.Dataset):
    def __init__(self, **task_args):
        self.task_args = task_args
        self.tr_val = task_args.get('tr_val', "train")
        self.root = task_args.get('root', "./dataset/pmemo")
        self.segment_type = task_args.get('segment_type', "all")
        self.cfg = task_args.get('cfg')

        # Path to the split file (train/val/test)
        self.split_file = os.path.join(self.root, 'meta', 'split', f"{self.tr_val}.txt")
        
        # Read file IDs from the split file
        with open(self.split_file, 'r') as f:
            self.file_ids = [line.strip() for line in f.readlines()]

        # Separate tonic and mode
        tonic_signatures = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        mode_signatures = ["major", "minor"]  # Major and minor modes

        self.tonic_to_idx = {tonic: idx for idx, tonic in enumerate(tonic_signatures)}
        self.mode_to_idx = {mode: idx for idx, mode in enumerate(mode_signatures)}

        self.idx_to_tonic = {idx: tonic for tonic, idx in self.tonic_to_idx.items()}
        self.idx_to_mode = {idx: mode for mode, idx in self.mode_to_idx.items()}

        with open('dataset/pmemo/meta/chord.json', 'r') as f:
            self.chord_to_idx = json.load(f)
        with open('dataset/pmemo/meta/chord_inv.json', 'r') as f:
            self.idx_to_chord = json.load(f)
            self.idx_to_chord = {int(k): v for k, v in self.idx_to_chord.items()}  # Ensure keys are ints
        with open('dataset/emomusic/meta/chord_root.json') as json_file:
            self.chordRootDic = json.load(json_file)
        with open('dataset/emomusic/meta/chord_attr.json') as json_file:
            self.chordAttrDic = json.load(json_file)            

        # MERT and MP3 directories
        self.mert_dir = os.path.join(self.root, 'mert_30s')
        self.mp3_dir = os.path.join(self.root, 'mp3')

        # Load static annotations (valence and arousal)
        self.annotation_file = os.path.join(self.root, 'meta', 'static_annotations.csv')
        self.annotations = pd.read_csv(self.annotation_file, index_col='song_id')

        # Load static annotations (valence and arousal)
        self.annotation_tag_file = os.path.join(self.root, 'meta', 'mood_probabilities.csv')
        self.annotations_tag = pd.read_csv(self.annotation_tag_file, index_col='song_id')

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_id = int(self.file_ids[index])  # File ID from split
        # Get valence and arousal from annotations
        if file_id not in self.annotations.index:
            raise ValueError(f"File ID {file_id} not found in annotations.")

        valence = self.annotations.loc[file_id, 'valence_mean']
        arousal = self.annotations.loc[file_id, 'arousal_mean']

        y_valence = torch.tensor(valence, dtype=torch.float32)
        y_arousal = torch.tensor(arousal, dtype=torch.float32)

        y_mood = np.array(self.annotations_tag.loc[file_id])
        y_mood = y_mood.astype('float32')
        y_mood = torch.from_numpy(y_mood)

        # --- Chord feature --- 
        fn_chord = os.path.join(self.root, 'chord', 'lab3', str(file_id) + ".lab")

        chords = []
        
        if not os.path.exists(fn_chord):
            chords.append((float(0), float(0), "N"))
        else:
            with open(fn_chord, 'r') as file:
                for line in file:
                    start, end, chord = line.strip().split()
                    chords.append((float(start), float(end), chord))

        encoded = []
        encoded_root= []
        encoded_attr=[]
        durations = []
        for start, end, chord in chords:
            chord_arr = chord.split(":")
            if len(chord_arr) == 1:
                chordRootID = self.chordRootDic[chord_arr[0]]
                if chord_arr[0] == "N" or chord_arr[0] == "X":
                    chordAttrID = 0
                else:
                    chordAttrID = 1
            elif len(chord_arr) == 2:
                chordRootID = self.chordRootDic[chord_arr[0]]
                chordAttrID = self.chordAttrDic[chord_arr[1]]
            encoded_root.append(chordRootID)
            encoded_attr.append(chordAttrID)

            if chord in self.chord_to_idx:
                encoded.append(self.chord_to_idx[chord])
            else:
                print(f"Warning: Chord {chord} not found in chord.json. Skipping.")
            
            durations.append(end - start)  # Compute duration
        
        encoded_chords = np.array(encoded)
        encoded_chords_root = np.array(encoded_root)
        encoded_chords_attr = np.array(encoded_attr)
        
        # Maximum sequence length for chords
        max_sequence_length = 100  # Define this globally or as a parameter

        # Truncate or pad chord sequences
        if len(encoded_chords) > max_sequence_length:
            # Truncate to max length
            encoded_chords = encoded_chords[:max_sequence_length]
            encoded_chords_root = encoded_chords_root[:max_sequence_length]
            encoded_chords_attr = encoded_chords_attr[:max_sequence_length]
        
        else:
            # Pad with zeros (padding value for chords)
            padding = [0] * (max_sequence_length - len(encoded_chords))
            encoded_chords = np.concatenate([encoded_chords, padding])
            encoded_chords_root = np.concatenate([encoded_chords_root, padding])
            encoded_chords_attr = np.concatenate([encoded_chords_attr, padding])
            
        # Convert to tensor
        chords_tensor = torch.tensor(encoded_chords, dtype=torch.long)  # Fixed length tensor
        chords_root_tensor = torch.tensor(encoded_chords_root, dtype=torch.long)  # Fixed length tensor
        chords_attr_tensor = torch.tensor(encoded_chords_attr, dtype=torch.long)  # Fixed length tensor

        # --- Key feature ---
        fn_key = os.path.join(self.root, 'key', str(file_id) + ".lab")

        if not os.path.exists(fn_key):
            mode = "major"
        else:
            mode = "major"  # Default value
            with open(fn_key, 'r') as file:
                for line in file:
                    key = line.strip()
            if key == "None":
                mode = "major"
            else:
                mode = key.split()[-1]
        
        encoded_mode = self.mode_to_idx.get(mode, 0)
        mode_tensor = torch.tensor([encoded_mode], dtype=torch.long)

        # --- MERT feature ---
        fn_mert = os.path.join(self.mert_dir, str(file_id))

        embeddings = []

        # Specify the layers to extract (3rd, 6th, 9th, and 12th layers)
        layers_to_extract = self.cfg.model.layers

        # Collect all segment embeddings
        segment_embeddings = []
        for filename in sorted(os.listdir(fn_mert)):  # Sort files to ensure sequential order
            file_path = os.path.join(fn_mert, filename)
            if os.path.isfile(file_path) and filename.endswith('.npy'):
                segment = np.load(file_path)

                # Extract and concatenate features for the specified layers
                concatenated_features = np.concatenate(
                    [segment[:, layer_idx, :] for layer_idx in layers_to_extract], axis=1
                )
                concatenated_features = np.squeeze(concatenated_features)  # Shape: 768 * 2 = 1536
                segment_embeddings.append(concatenated_features)

        # Convert to numpy array
        segment_embeddings = np.array(segment_embeddings)

        # Check mode: 'train' or 'val'
        if self.tr_val == "train" and len(segment_embeddings) > 0:  # Augmentation for training
            num_segments = len(segment_embeddings)
            
            # Randomly choose a starting index and the length of the sequence
            start_idx = np.random.randint(0, num_segments)  # Random starting index
            end_idx = np.random.randint(start_idx + 1, num_segments + 1)  # Ensure end index is after start index

            # Extract the sequential subset
            chosen_segments = segment_embeddings[start_idx:end_idx]

            # Compute the mean of the chosen sequential segments
            final_embedding_mert = np.mean(chosen_segments, axis=0)
        else:  # Validation or other modes: Use mean of all segments
            if len(segment_embeddings) > 0:
                final_embedding_mert = np.mean(segment_embeddings, axis=0)
            else:
                # Handle case with no valid embeddings
                final_embedding_mert = np.zeros((1536,))  # Example: Return zero vector of appropriate size

        # Convert to PyTorch tensor
        final_embedding_mert = torch.from_numpy(final_embedding_mert)
        
        # Get the MP3 path
        mp3_path = os.path.join(self.mp3_dir, f"{file_id}.mp3")
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"MP3 file not found for {mp3_path}")

        return {
            "x_mert": final_embedding_mert,
            "x_chord" : chords_tensor,
            "x_chord_root" : chords_root_tensor,
            "x_chord_attr" : chords_attr_tensor,
            "x_key" : mode_tensor,
            "y_va": torch.stack([y_valence, y_arousal], dim=0),
            "y_mood" : y_mood,
            "path": mp3_path
        }