# utils/calculations.py
"""
Business Calculations: Cost, CO, Savings
"""

def calculate_financial_impact(energy_saved_kwh, electricity_rate=None):
    """Calculate cost savings from energy reduction"""
    from utils.config import BUSINESS_SETTINGS
    rate = electricity_rate or BUSINESS_SETTINGS['electricity_rate']
    
    daily_savings = energy_saved_kwh * rate
    monthly_savings = daily_savings * BUSINESS_SETTINGS['working_days_per_month']
    annual_savings = daily_saved_kwh * BUSINESS_SETTINGS['working_days_per_year']
    
    return {
        'daily': round(daily_savings, 2),
        'monthly': round(monthly_savings, 2),
        'annual': round(annual_savings, 2),
        'energy_saved': energy_saved_kwh,
        'rate': rate
    }

def calculate_environmental_impact(energy_saved_kwh):
    """Calculate CO reduction from energy savings"""
    from utils.config import BUSINESS_SETTINGS
    
    co2_saved = energy_saved_kwh * BUSINESS_SETTINGS['co2_per_kwh']
    trees_equivalent = co2_saved / BUSINESS_SETTINGS['tree_co2_absorption']
    
    return {
        'co2_kg': round(co2_saved, 2),
        'trees_equivalent': round(trees_equivalent, 2),
        'monthly_co2': round(co2_saved * 30, 2),
        'annual_co2': round(co2_saved * 365, 2)
    }

def calculate_optimization_potential(current_energy, optimization_percentage=0.15):
    """Calculate potential energy savings"""
    optimized_energy = current_energy * (1 - optimization_percentage)
    energy_saved = current_energy - optimized_energy
    
    financial = calculate_financial_impact(energy_saved)
    environmental = calculate_environmental_impact(energy_saved)
    
    return {
        'current_energy': round(current_energy, 2),
        'optimized_energy': round(optimized_energy, 2),
        'energy_saved': round(energy_saved, 2),
        'savings_percentage': optimization_percentage * 100,
        'financial': financial,
        'environmental': environmental
    }
