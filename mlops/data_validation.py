# mlops/data_validation.py
import pandas as pd
import numpy as np
import os

def validate_data_quality():
    """Validate data quality for MLOps pipeline"""
    print(" Validating data quality...")
    
    data_path = "data/raw/research_based_campus_energy.csv"
    
    if not os.path.exists(data_path):
        print(" Data file not found")
        return False
    
    try:
        df = pd.read_csv(data_path)
        
        # Basic checks
        print(f" Dataset shape: {df.shape}")
        print(f" Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f" Null values found:\n{null_counts[null_counts > 0]}")
        
        # Check energy values
        if 'energy_kwh' in df.columns:
            energy_stats = df['energy_kwh'].describe()
            print(f" Energy statistics:\n{energy_stats}")
            
            # Check for unrealistic values
            unrealistic = df[(df['energy_kwh'] <= 0) | (df['energy_kwh'] > 1000)]
            if len(unrealistic) > 0:
                print(f" {len(unrealistic)} unrealistic energy values found")
        
        print(" Data validation completed")
        return True
        
    except Exception as e:
        print(f" Data validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_data_quality()
