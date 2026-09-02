"""Sliding-window CNN-LSTM inference, one buffer per tracked person."""

import os
from collections import defaultdict, deque

import numpy as np
import torch

from src.classification.train_cnn_lstm import CNNLSTMClassifier
from normalize_sequences import normalize_sequences

WINDOW_SIZE = 15
CLASS_NAMES = ['normal', 'fall', 'climb']
MODEL_PATH = "models/saved/child_cnn_lstm_best.pth"


class ActivityClassifier:

    def __init__(self, model_path=MODEL_PATH, device=None, window_size=WINDOW_SIZE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train it first, or pass model_path."
            )

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.window_size = window_size

        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint.get('class_names', CLASS_NAMES)

        saved_window = checkpoint.get('window_size')
        if saved_window and saved_window != window_size:
            raise ValueError(
                f"Model expects window_size={saved_window} but {window_size} requested"
            )

        # Read from the checkpoint rather than the module default, so a model
        # trained on a different feature count still loads
        self.input_size = checkpoint.get('input_size', 51)
        self.model = CNNLSTMClassifier(input_size=self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.buffers = defaultdict(lambda: deque(maxlen=window_size))

        print(f"  ActivityClassifier: {model_path} "
              f"(epoch {checkpoint.get('epoch', '?')}, "
              f"val acc {checkpoint.get('val_accuracy', 0):.3f}) on {self.device}")

    def predict(self, track_id, feature_vector):
        """Returns None until this track has window_size frames buffered."""
        self.buffers[track_id].append(np.asarray(feature_vector, dtype=np.float32))

        if len(self.buffers[track_id]) < self.window_size:
            return None

        window = np.array(self.buffers[track_id], dtype=np.float32)[None, :, :]

        # The model was trained on person-relative coordinates, but
        # compute_features() returns raw pixels — this must match training
        window = normalize_sequences(window)

        with torch.no_grad():
            logits = self.model(torch.from_numpy(window).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        return {
            'activity': self.class_names[idx],
            'confidence': float(probs[idx]),
            'probabilities': {n: float(p) for n, p in zip(self.class_names, probs)},
        }

    def buffer_progress(self, track_id):
        return len(self.buffers[track_id]), self.window_size

    def cleanup(self, active_track_ids):
        """Drop buffers for people who have left, so memory does not grow."""
        active = set(active_track_ids)
        for tid in [t for t in self.buffers if t not in active]:
            del self.buffers[tid]

    def reset(self):
        self.buffers.clear()
