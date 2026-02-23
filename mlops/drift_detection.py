# mlops/drift_detection.py
import pandas as pd
import numpy as np
import json
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import os
import sys

# Attempt to import Evidently – fallback if anything fails
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
    EVIDENTLY_AVAILABLE = True
except Exception as e:
    EVIDENTLY_AVAILABLE = False

class DriftDetector:
    def __init__(self, reference_data=None):
        self.reference_data = reference_data
        self.drift_history = []

    def load_reference_data(self, path='data/reference/reference_data.csv'):
        """Load or create reference dataset (e.g., training data snapshot)"""
        if os.path.exists(path):
            self.reference_data = pd.read_csv(path)
            print(f"✅ Reference data loaded: {len(self.reference_data)} samples")
        else:
            # Create reference from existing research data
            df = pd.read_csv('data/raw/research_based_campus_energy.csv')
            self.reference_data = df.sample(1000, random_state=42)
            os.makedirs('data/reference', exist_ok=True)
            self.reference_data.to_csv(path, index=False)
            print(f"✅ Reference data created and saved to {path}")

    def check_drift_simple(self, current_df, threshold=0.1):
        """Simple statistical drift detection (fallback)"""
        if self.reference_data is None:
            self.load_reference_data()

        results = {}
        # Check numerical features
        numerical_features = ['temperature', 'occupancy', 'hour_of_day']
        for col in numerical_features:
            if col not in self.reference_data or col not in current_df:
                continue
            ref_mean = self.reference_data[col].mean()
            cur_mean = current_df[col].mean()
            rel_change = abs(cur_mean - ref_mean) / ref_mean if ref_mean != 0 else 0
            drift_detected = bool(rel_change > threshold)
            results[col] = {
                'ref_mean': round(ref_mean, 3),
                'cur_mean': round(cur_mean, 3),
                'relative_change': round(rel_change, 3),
                'drift_detected': drift_detected
            }

        # Check target drift
        if 'energy_kwh' in current_df.columns:
            ref_target = self.reference_data['energy_kwh'].mean()
            cur_target = current_df['energy_kwh'].mean()
            target_change = abs(cur_target - ref_target) / ref_target
            drift_detected = bool(target_change > threshold)
            results['target'] = {
                'ref_mean': round(ref_target, 3),
                'cur_mean': round(cur_target, 3),
                'relative_change': round(target_change, 3),
                'drift_detected': drift_detected
            }

        # Overall drift flag
        overall_drift = any(r.get('drift_detected', False) for r in results.values())
        return {'drift_detected': overall_drift, 'details': results}

    def check_drift_evidently(self, current_df, column_mapping=None):
        """Use Evidently for comprehensive drift detection"""
        if not EVIDENTLY_AVAILABLE:
            return self.check_drift_simple(current_df)

        if self.reference_data is None:
            self.load_reference_data()

        if column_mapping is None:
            column_mapping = ColumnMapping()
            column_mapping.numerical_features = ['temperature', 'occupancy', 'hour_of_day']
            column_mapping.target = 'energy_kwh'

        drift_report = Report(metrics=[
            ColumnDriftMetric(column_name='temperature'),
            ColumnDriftMetric(column_name='occupancy'),
            ColumnDriftMetric(column_name='hour_of_day'),
            DatasetDriftMetric()
        ])

        drift_report.run(reference_data=self.reference_data,
                        current_data=current_df,
                        column_mapping=column_mapping)

        result = drift_report.as_dict()
        # Extract drift status
        drift_detected = any(
            metric['result']['drift_detected']
            for metric in result['metrics']
            if 'drift_detected' in metric['result']
        )
        return {
            'drift_detected': drift_detected,
            'report': result,
            'timestamp': datetime.now().isoformat()
        }

    def check_performance_drift(self, model, X_test, y_test, threshold=0.15):
        """Check if model performance degraded"""
        from sklearn.metrics import mean_absolute_error
        y_pred = model.predict(X_test)
        current_mae = mean_absolute_error(y_test, y_pred)

        # Get baseline MAE from MLflow production model
        try:
            client = MlflowClient()
            mlflow.set_tracking_uri("http://localhost:5000")  # adjust as needed
            model_name = "EnergyPredictor1"
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            if prod_versions:
                run = client.get_run(prod_versions[0].run_id)
                baseline_mae = run.data.metrics.get("mae", None)
                if baseline_mae:
                    degradation = (current_mae - baseline_mae) / baseline_mae
                    drift = bool(degradation > threshold)
                    return {
                        'drift_detected': drift,
                        'current_mae': current_mae,
                        'baseline_mae': baseline_mae,
                        'degradation': round(degradation, 3),
                        'threshold': threshold
                    }
        except Exception as e:
            print(f"Could not get baseline MAE: {e}")

        return {'drift_detected': False, 'note': 'Baseline MAE not available'}

    def run_check(self, current_data_path, output_path='drift_report.json'):
        """Run drift check on new data and save report"""
        if not os.path.exists(current_data_path):
            print(f"❌ File not found: {current_data_path}")
            return
        current_df = pd.read_csv(current_data_path)
        result = self.check_drift_evidently(current_df) if EVIDENTLY_AVAILABLE else self.check_drift_simple(current_df)

        # Save history
        self.drift_history.append(result)
        with open(output_path, 'w') as f:
            json.dump({
                'last_check': result,
                'history': self.drift_history[-10:]
            }, f, indent=2)

        if result['drift_detected']:
            print("🚨 Drift detected! Consider retraining.")
        else:
            print("✅ No significant drift detected.")

        return result

if __name__ == "__main__":
    detector = DriftDetector()
    detector.load_reference_data()
    data_path = 'data/raw/research_based_campus_energy.csv'  # or use current data
    if not os.path.exists(data_path):
        print(f"❌ Data file {data_path} not found")
        sys.exit(1)
    result = detector.run_check(data_path)
    print(json.dumps(result, indent=2))
    # Exit with error if drift detected
    if result.get('drift_detected', False):
        sys.exit(1)
    else:
        sys.exit(0)