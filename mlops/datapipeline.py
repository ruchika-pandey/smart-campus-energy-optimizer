# mlops/data_pipeline.py
"""
Data Collection and Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EnergyDataPipeline:
    def __init__(self):
        self.data = None
        
    def generate_sample_data(self, days=30, buildings=None):
        """Generate synthetic energy data for demo"""
        if buildings is None:
            buildings = ['Library', 'Lab', 'Classroom', 'Hostel']
        
        np.random.seed(42)
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            date = start_date + timedelta(days=day)
            for hour in range(24):
                for building in buildings:
                    # Base pattern with daily/weekly cycles
                    base_energy = 100 + 20 * np.sin(2 * np.pi * hour / 24)
                    
                    # Building-specific adjustments
                    building_factors = {
                        'Library': 1.2,
                        'Lab': 1.5,
                        'Classroom': 1.0,
                        'Hostel': 0.8
                    }
                    
                    # Day of week effect
                    day_of_week = date.weekday()
                    if day_of_week >= 5:  # Weekend
                        base_energy *= 0.6
                    
                    # Add randomness
                    energy = base_energy * building_factors.get(building, 1.0)
                    energy += np.random.normal(0, 10)
                    energy = max(20, energy)  # Minimum threshold
                    
                    # Occupancy simulation
                    occupancy_base = 0.3 + 0.4 * np.sin(2 * np.pi * hour / 24)
                    occupancy = min(0.95, max(0.05, occupancy_base + np.random.normal(0, 0.1)))
                    
                    # Temperature simulation
                    temp = 25 + 8 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 3)
                    
                    data.append({
                        'timestamp': date.replace(hour=hour, minute=0, second=0),
                        'building': building,
                        'energy_kwh': round(energy, 2),
                        'occupancy': round(occupancy, 2),
                        'temperature': round(temp, 1),
                        'day_of_week': day_of_week,
                        'hour_of_day': hour,
                        'is_weekend': 1 if day_of_week >= 5 else 0,
                        'is_peak_hour': 1 if 14 <= hour <= 18 else 0
                    })
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def preprocess_for_lstm(self, building_name=None):
        """Prepare data for LSTM training"""
        if self.data is None:
            raise ValueError("No data available. Generate or load data first.")
        
        if building_name:
            building_data = self.data[self.data['building'] == building_name].copy()
        else:
            building_data = self.data.copy()
        
        # Sort by timestamp
        building_data = building_data.sort_values('timestamp')
        
        # Feature selection
        features = building_data[['energy_kwh', 'temperature', 'occupancy']].values
        target = building_data['energy_kwh'].values
        
        return features, target, building_data
    
    def create_sequences(self, data, sequence_length=24):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, 0])  # Predict energy only
        return np.array(X), np.array(y)
