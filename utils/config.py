# utils/config.py
"""
Project Configuration Settings
"""

# Data Settings
DATA_SETTINGS = {
    'sequence_length': 24,           # Hours to use for prediction
    'train_test_split': 0.2,         # Test set percentage
    'validation_split': 0.2,         # Validation percentage
    'target_column': 'energy_kwh',   # What we're predicting
}

# Model Settings
MODEL_SETTINGS = {
    'lstm_units': [50, 50],          # LSTM layer sizes
    'dense_units': [25],             # Dense layer sizes
    'dropout_rate': 0.2,             # Dropout percentage
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
}

# Business Settings
BUSINESS_SETTINGS = {
    'electricity_rate': 8.0,         # ₹ per kWh
    'co2_per_kwh': 0.82,             # kg CO₂ per kWh
    'tree_co2_absorption': 20,       # kg CO₂ per tree per year
    'working_days_per_month': 30,
    'working_days_per_year': 365,
}

# File Paths
PATHS = {
    'data_raw': 'data/raw/',
    'data_processed': 'data/processed/',
    'models': 'models/saved_models/',
    'reports': 'reports/',
    'visualizations': 'visualizations/',
}