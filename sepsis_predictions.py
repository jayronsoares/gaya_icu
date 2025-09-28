import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# SIRS Criteria Based Sepsis Risk Assessment
def calculate_sirs_score(vitals: Dict) -> int:
    """Calculate SIRS (Systemic Inflammatory Response Syndrome) score"""
    score = 0
    
    # Temperature criteria (>38°C or <36°C)
    temp = vitals.get('temperature', 37.0)
    if temp > 38.0 or temp < 36.0:
        score += 1
    
    # Heart rate >90 bpm
    hr = vitals.get('heart_rate', 80)
    if hr > 90:
        score += 1
    
    # Respiratory rate >20/min
    rr = vitals.get('respiratory_rate', 16)
    if rr > 20:
        score += 1
    
    # White blood cell count criteria - only count if available
    wbc = vitals.get('white_blood_cells')
    if wbc is not None and (wbc > 12.0 or wbc < 4.0):
        score += 1
    
    return score

def calculate_sepsis_probability(vitals: Dict, lab_results: pd.DataFrame = None) -> Tuple[float, str, List[str]]:
    """
    Calculate sepsis probability using multiple clinical indicators
    Returns: (probability_percentage, status, risk_factors)
    """
    base_probability = 0.0
    risk_factors = []
    
    # Integrate WBC from lab results if available
    vitals_with_wbc = vitals.copy()
    if lab_results is not None and not lab_results.empty:
        wbc_labs = lab_results[lab_results['test_name'] == 'White Blood Cells']
        if not wbc_labs.empty:
            wbc_value = wbc_labs.iloc[0]['test_value']
            vitals_with_wbc['white_blood_cells'] = wbc_value
    
    # SIRS Score Component (0-40% based on SIRS criteria)
    sirs_score = calculate_sirs_score(vitals_with_wbc)
    sirs_probability = min(sirs_score * 10, 40)  # Max 40% from SIRS
    base_probability += sirs_probability
    
    if sirs_score >= 2:
        risk_factors.append(f"SIRS criteria met ({sirs_score}/4 points)")
    
    # Vital Signs Risk Factors
    temp = vitals.get('temperature', 37.0)
    if temp > 38.5:
        base_probability += 15
        risk_factors.append(f"High fever ({temp}°C)")
    elif temp < 35.5:
        base_probability += 20
        risk_factors.append(f"Hypothermia ({temp}°C)")
    
    # Hypotension (strong sepsis indicator)
    sbp = vitals.get('blood_pressure_systolic', 120)
    if sbp < 90:
        base_probability += 25
        risk_factors.append(f"Hypotension ({sbp} mmHg)")
    elif sbp < 100:
        base_probability += 10
        risk_factors.append(f"Low blood pressure ({sbp} mmHg)")
    
    # Oxygen saturation
    spo2 = vitals.get('oxygen_saturation', 98)
    if spo2 < 90:
        base_probability += 20
        risk_factors.append(f"Severe hypoxemia ({spo2}%)")
    elif spo2 < 95:
        base_probability += 10
        risk_factors.append(f"Hypoxemia ({spo2}%)")
    
    # Tachycardia with fever (combination risk)
    hr = vitals.get('heart_rate', 80)
    if hr > 120 and temp > 38.0:
        base_probability += 15
        risk_factors.append("Tachycardia with fever")
    elif hr > 100:
        base_probability += 5
        risk_factors.append(f"Tachycardia ({hr} bpm)")
    
    # Lab results enhancement (if available)
    if lab_results is not None and not lab_results.empty:
        lab_probability, lab_risks = analyze_lab_results_for_sepsis(lab_results)
        base_probability += lab_probability
        risk_factors.extend(lab_risks)
    
    # Cap probability at 95%
    final_probability = min(base_probability, 95.0)
    
    # Determine status based on probability
    if final_probability < 30:
        status = 'stable'
    elif final_probability < 60:
        status = 'alert'
    else:
        status = 'critical'
    
    return final_probability, status, risk_factors

def analyze_lab_results_for_sepsis(lab_results: pd.DataFrame) -> Tuple[float, List[str]]:
    """Analyze lab results for sepsis indicators"""
    lab_probability = 0.0
    lab_risks = []
    
    for _, lab in lab_results.iterrows():
        test_name = lab['test_name']
        test_value = lab['test_value']
        
        # Lactate (strong sepsis indicator)
        if test_name == 'Lactate' and test_value > 2.0:
            if test_value > 4.0:
                lab_probability += 20
                lab_risks.append(f"Very high lactate ({test_value} mmol/L)")
            else:
                lab_probability += 10
                lab_risks.append(f"Elevated lactate ({test_value} mmol/L)")
        
        # Procalcitonin (sepsis biomarker)
        elif test_name == 'Procalcitonin' and test_value > 0.25:
            if test_value > 2.0:
                lab_probability += 25
                lab_risks.append(f"Very high procalcitonin ({test_value} ng/mL)")
            elif test_value > 0.5:
                lab_probability += 15
                lab_risks.append(f"High procalcitonin ({test_value} ng/mL)")
            else:
                lab_probability += 8
                lab_risks.append(f"Elevated procalcitonin ({test_value} ng/mL)")
        
        # C-Reactive Protein
        elif test_name == 'C-Reactive Protein' and test_value > 100:
            lab_probability += 10
            lab_risks.append(f"Very high CRP ({test_value} mg/L)")
        elif test_name == 'C-Reactive Protein' and test_value > 50:
            lab_probability += 5
            lab_risks.append(f"High CRP ({test_value} mg/L)")
        
        # White Blood Cells
        elif test_name == 'White Blood Cells':
            if test_value > 15 or test_value < 4:
                lab_probability += 8
                lab_risks.append(f"Abnormal WBC ({test_value} x10³/μL)")
    
    return lab_probability, lab_risks

def predict_sepsis_onset_time(current_probability: float, vitals_trend: pd.DataFrame) -> Tuple[int, str]:
    """
    Predict time to sepsis onset based on current probability and vital trends
    Returns: (hours_to_onset, risk_level)
    """
    if current_probability < 30:
        return 48, "low_risk"
    elif current_probability < 60:
        # Analyze vital sign trends
        if not vitals_trend.empty and len(vitals_trend) > 3:
            # Calculate trend in temperature and heart rate
            temp_trend = np.polyfit(range(len(vitals_trend)), vitals_trend['temperature'], 1)[0]
            hr_trend = np.polyfit(range(len(vitals_trend)), vitals_trend['heart_rate'], 1)[0]
            
            if temp_trend > 0.1 and hr_trend > 2:  # Rising temp and HR
                return 12, "moderate_risk"
            elif temp_trend > 0.05 or hr_trend > 1:
                return 24, "moderate_risk"
            else:
                return 36, "moderate_risk"
        else:
            return 24, "moderate_risk"
    else:
        return 6, "high_risk"  # Immediate risk

# Advanced sepsis prediction using multiple factors
def advanced_sepsis_prediction(patient_data: Dict, vitals_history: pd.DataFrame, lab_results: pd.DataFrame) -> Dict:
    """
    Advanced sepsis prediction combining multiple data sources
    """
    # Get current vitals from most recent vital signs
    if not vitals_history.empty:
        latest_vitals = vitals_history.iloc[0]
        current_vitals = {
            'temperature': latest_vitals.get('temperature', 37.0),
            'heart_rate': latest_vitals.get('heart_rate', 80),
            'respiratory_rate': latest_vitals.get('respiratory_rate', 16),
            'oxygen_saturation': latest_vitals.get('oxygen_saturation', 98),
            'blood_pressure_systolic': latest_vitals.get('blood_pressure_systolic', 120)
        }
    else:
        # Fallback to defaults if no vitals history
        current_vitals = {
            'temperature': patient_data.get('temperature', 37.0),
            'heart_rate': patient_data.get('heart_rate', 80),
            'respiratory_rate': patient_data.get('respiratory_rate', 16),
            'oxygen_saturation': patient_data.get('oxygen_saturation', 98),
            'blood_pressure_systolic': patient_data.get('blood_pressure_systolic', 120)
        }
    
    # Calculate current sepsis probability
    probability, status, risk_factors = calculate_sepsis_probability(current_vitals, lab_results)
    
    # Predict onset time
    onset_hours, risk_level = predict_sepsis_onset_time(probability, vitals_history)
    
    # Calculate confidence based on data availability
    confidence = 0.7  # Base confidence
    if not lab_results.empty:
        confidence += 0.2
    if len(vitals_history) > 12:  # More than 12 hours of data
        confidence += 0.1
    
    confidence = min(confidence, 0.95)
    
    return {
        'sepsis_probability': probability,
        'status': status,
        'risk_factors': risk_factors,
        'predicted_onset_hours': onset_hours,
        'risk_level': risk_level,
        'confidence': confidence,
        'sirs_score': calculate_sirs_score(current_vitals)
    }
