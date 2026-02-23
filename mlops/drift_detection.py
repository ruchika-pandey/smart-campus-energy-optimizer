# mlops/drift_detection.py
import pandas as pd
import numpy as np
import json
import os
import sys

class DriftDetector:
    def __init__(self, reference_data=None):
        self.reference_data = reference_data

    def load_reference_data(self, path='data/reference/reference_data.csv'):
        if os.path.exists(path):
            self.reference_data = pd.read_csv(path)
            print(f"[OK] Reference data loaded: {len(self.reference_data)} samples")
        else:
            # Create synthetic reference data
            np.random.seed(42)
            n = 1000
            df = pd.DataFrame({
                'hour_of_day': np.random.randint(0, 24, n),
                'temperature': np.random.uniform(15, 35, n),
                'occupancy': np.random.uniform(0.1, 0.9, n),
                'energy_kwh': np.random.uniform(50, 300, n)
            })
            self.reference_data = df
            os.makedirs('data/reference', exist_ok=True)
            self.reference_data.to_csv(path, index=False)
            print(f"[OK] Reference data created and saved to {path}")

    def load_current_data(self, path='data/raw/research_based_campus_energy.csv'):
        """Load current data, create synthetic if missing"""
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            print(f"[WARN] Data file {path} not found. Using synthetic data.")
            np.random.seed(43)  # Different seed for variation
            n = 500
            return pd.DataFrame({
                'hour_of_day': np.random.randint(0, 24, n),
                'temperature': np.random.uniform(15, 35, n),
                'occupancy': np.random.uniform(0.1, 0.9, n),
                'energy_kwh': np.random.uniform(50, 300, n)
            })

    def check_drift_simple(self, current_df, threshold=0.1):
        if self.reference_data is None:
            self.load_reference_data()
        results = {}
        for col in ['temperature', 'occupancy', 'hour_of_day']:
            if col in self.reference_data and col in current_df:
                ref_mean = self.reference_data[col].mean()
                cur_mean = current_df[col].mean()
                rel_change = abs(cur_mean - ref_mean) / (ref_mean if ref_mean != 0 else 1)
                results[col] = {
                    'ref_mean': round(ref_mean, 3),
                    'cur_mean': round(cur_mean, 3),
                    'relative_change': round(rel_change, 3),
                    'drift_detected': bool(rel_change > threshold)
                }
        if 'energy_kwh' in current_df.columns:
            ref = self.reference_data['energy_kwh'].mean()
            cur = current_df['energy_kwh'].mean()
            change = abs(cur - ref) / (ref if ref != 0 else 1)
            results['target'] = {
                'ref_mean': round(ref, 3),
                'cur_mean': round(cur, 3),
                'relative_change': round(change, 3),
                'drift_detected': bool(change > threshold)
            }
        overall_drift = any(r.get('drift_detected', False) for r in results.values())
        return {'drift_detected': overall_drift, 'details': results}

    def run_check(self, current_data_path, output_path='drift_report.json'):
        current_df = self.load_current_data(current_data_path)
        result = self.check_drift_simple(current_df)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        if result['drift_detected']:
            print("[DRIFT] Drift detected! Consider retraining.")
        else:
            print("[OK] No significant drift detected.")
        return result

if __name__ == "__main__":
    detector = DriftDetector()
    detector.load_reference_data()
    result = detector.run_check('data/raw/research_based_campus_energy.csv')
    print(json.dumps(result, indent=2))
    sys.exit(1 if result.get('drift_detected') else 0)