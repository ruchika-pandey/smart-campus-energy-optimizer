# tests/test_calculations.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.calculations import calculate_optimization_potential

def test_calculate_optimization_potential():
    """Test optimization potential calculation"""
    result = calculate_optimization_potential(100)
    
    assert result['current_energy'] == 100
    assert result['optimized_energy'] == 85  # 15% less
    assert result['energy_saved'] == 15
    assert result['savings_percentage'] == 15.0
    
    # Financial calculations
    assert result['financial']['daily'] == 15 * 8  # ₹8 per kWh
    
    print("✅ All calculation tests passed")

if __name__ == "__main__":
    test_calculate_optimization_potential()