"""
Activity Classifier — combines LSTM predictions with rule-based detection.
This is the module that makes the final activity decision for each person.
"""

import torch
import numpy as np
from collections import deque


class ActivityClassifier:
    """
    Classifies activities for each tracked person using both:
    1. LSTM model (learned patterns from training data)
    2. Rule-based detector (geometric thresholds as safety net)
    """
    
    CLASS_NAMES = ['fall', 'climb', 'fight', 'normal']
    
    def __init__(self, model_path="models/saved/activity_lstm_best.pth",
                 window_size=30, device="cuda"):
        """
        Args:
            model_path: Path to trained LSTM weights
            window_size: Frames per classification window
            device: "cuda" or "cpu"
        """
        from src.classification.train_lstm import ActivityLSTM
        
        self.window_size = window_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load trained model
        self.model = ActivityLSTM(
            input_size=99,
            hidden_size=128,
            num_layers=2,
            num_classes=len(self.CLASS_NAMES)
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Per-person landmark history: {track_id: deque}
        self.landmark_history = {}
    
    def update_and_classify(self, track_id, landmark_vector):
        """
        Add new frame landmarks for a person and classify if enough history.
        
        Args:
            track_id: Person tracking ID
            landmark_vector: List of 99 values (33 landmarks × 3 coords)
            
        Returns:
            dict with 'class', 'confidence', 'probabilities' or None if not enough data
        """
        if track_id not in self.landmark_history:
            self.landmark_history[track_id] = deque(maxlen=self.window_size)
        
        self.landmark_history[track_id].append(landmark_vector)
        
        # Need full window to classify
        if len(self.landmark_history[track_id]) < self.window_size:
            return None
        
        # Create input tensor
        sequence = np.array(list(self.landmark_history[track_id]), dtype=np.float32)
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
        
        return {
            'class': self.CLASS_NAMES[predicted_class],
            'class_idx': predicted_class,
            'confidence': confidence,
            'probabilities': {
                name: float(prob) 
                for name, prob in zip(self.CLASS_NAMES, probabilities)
            }
        }
    
    def cleanup(self, active_track_ids):
        """Remove history for persons no longer tracked."""
        old_ids = [tid for tid in self.landmark_history if tid not in active_track_ids]
        for tid in old_ids:
            del self.landmark_history[tid]