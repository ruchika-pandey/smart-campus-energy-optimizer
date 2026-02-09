# synthetic_generator.py - Keep for demos and testing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class SyntheticDataGenerator:
    """Generate synthetic data for quick demos and testing"""
    
    def __init__(self):
        self.data_dir = "data/synthetic"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def generate_quick_demo_data(self, days=7):
        """Generate quick demo data for presentations"""
        print("⚡ Generating quick demo data...")
        
        np.random.seed(42)
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            date = start_date + timedelta(days=day)
            for hour in range(24):
                # Base pattern
                base = 100 + 20 * np.sin(2 * np.pi * hour / 24)
                
                # Day effect
                if date.weekday() >= 5:
                    base *= 0.7  # Weekend
                
                # Add noise
                energy = base + np.random.normal(0, 10)
                energy = max(50, min(energy, 200))
                
                data.append({
                    'timestamp': date.replace(hour=hour),
                    'building': 'Library',
                    'energy_kwh': round(energy, 2),
                    'temperature': round(25 + 5 * np.sin(2 * np.pi * hour / 24), 1),
                    'occupancy': round(0.3 + 0.5 * np.sin(2 * np.pi * (hour - 10) / 16), 2),
                    'hour': hour,
                    'day_of_week': date.weekday(),
                    'is_weekend': 1 if date.weekday() >= 5 else 0
                })
        
        df = pd.DataFrame(data)
        file_path = os.path.join(self.data_dir, "quick_demo_data.csv")
        df.to_csv(file_path, index=False)
        
        print(f"✅ Quick demo data saved: {file_path}")
        print(f"   • Records: {len(df):,}")
        print(f"   • Days: {days}")
        
        return df
    
    def generate_test_dataset(self):
        """Generate test dataset for model validation"""
        print("🧪 Generating test dataset...")
        
        # Similar to research data but smaller
        return self.generate_quick_demo_data(days=30)