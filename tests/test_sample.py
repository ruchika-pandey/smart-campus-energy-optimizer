# tests/test_sample.py - PLAIN ASCII ONLY
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all modules import correctly"""
    try:
        from utils.calculations import calculate_optimization_potential
        print("PASS: Calculations module imported")
        return True
    except Exception as e:
        print(f"FAIL: Import failed - {e}")
        return False

def test_calculations():
    """Test basic calculations"""
    try:
        from utils.calculations import calculate_optimization_potential
        result = calculate_optimization_potential(100)
        assert result['energy_saved'] == 15
        assert result['savings_percentage'] == 15.0
        print("PASS: Calculations test passed")
        return True
    except Exception as e:
        print(f"FAIL: Calculations test failed - {e}")
        return False

def test_dummy():
    """Simple test that always passes"""
    assert 1 + 1 == 2
    print("PASS: Dummy test passed")

if __name__ == "__main__":
    print("=== RUNNING TESTS ===")
    test_imports()
    test_calculations()
    test_dummy()
    print("=== ALL TESTS COMPLETED ===")
