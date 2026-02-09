# models/lstm_model.py - SIMPLIFIED
"""
Simulated LSTM Model for Demo
"""

import numpy as np

class SimpleEnergyPredictor:
    """Simple predictor for demo (no TensorFlow)"""
    
    def __init__(self):
        self.trained = False
        
    def train(self, data):
        """Simulate training"""
        print("🧠 Training simulated model...")
        self.trained = True
        return {"accuracy": 0.92, "loss": 0.08}
    
    def predict(self, conditions):
        """Simple prediction based on rules"""
        base = 100 + conditions.get('hour', 12) * 8
        
        # Adjustments
        if conditions.get('temperature', 25) > 28:
            base *= 1.3
        if conditions.get('occupancy', 0.5) > 0.7:
            base *= 1.2
        if conditions.get('special_event', False):
            base *= 1.4
        if conditions.get('is_weekend', False):
            base *= 0.7
            
        return round(base, 2)