import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

def calculate_base_los_score(patient_data: Dict) -> int:
    """Calculate base length of stay score based on patient demographics and diagnosis"""
    base_score = 3  # Minimum ICU stay
    
    # Age factor
    age = patient_data.get('age', 50)
    if age > 80:
        base_score += 4
    elif age > 65:
        base_score += 2
    elif age > 50:
        base_score += 1
    
    # Diagnosis-based scoring
    diagnosis = patient_data.get('diagnosis', '').lower()
    
    diagnosis_scores = {
        'septic shock': 12,
        'multi-organ failure': 15,
        'cardiac arrest': 10,
        'respiratory failure': 8,
        'pneumonia': 6,
        'stroke': 7,
        'kidney failure': 9,
        'post-surgical': 4,
        'diabetic ketoacidosis': 5
    }
    
    for condition, score in diagnosis_scores.items():
        if condition in diagnosis:
            base_score += score
            break
    else:
        base_score += 5  # Default for unspecified conditions
    
    return base_score

def calculate_severity_multiplier(vitals: Dict, lab_results: pd.DataFrame = None) -> float:
    """Calculate severity multiplier based on current patient condition"""
    multiplier = 1.0
    
    # Vital signs severity
    temp = vitals.get('temperature', 37.0)
    if temp > 39.0 or temp < 35.0:
        multiplier += 0.3
    elif temp > 38.5 or temp < 36.0:
        multiplier += 0.15
    
    # Blood pressure
    sbp = vitals.get('blood_pressure_systolic', 120)
    if sbp < 90:
        multiplier += 0.4  # Hypotension significantly extends stay
    elif sbp < 100:
        multiplier += 0.2
    elif sbp > 180:
        multiplier += 0.15
    
    # Oxygen saturation
    spo2 = vitals.get('oxygen_saturation', 98)
    if spo2 < 90:
        multiplier += 0.5
    elif spo2 < 95:
        multiplier += 0.25
    
    # Heart rate
    hr = vitals.get('heart_rate', 80)
    if hr > 120 or hr < 50:
        multiplier += 0.2
    elif hr > 100 or hr < 60:
        multiplier += 0.1
    
    # Lab results severity (if available)
    if lab_results is not None and not lab_results.empty:
        lab_multiplier = calculate_lab_severity_multiplier(lab_results)
        multiplier += lab_multiplier
    
    return min(multiplier, 3.0)  # Cap at 3x multiplier

def calculate_lab_severity_multiplier(lab_results: pd.DataFrame) -> float:
    """Calculate severity multiplier based on lab results"""
    lab_multiplier = 0.0
    
    for _, lab in lab_results.iterrows():
        test_name = lab['test_name']
        test_value = lab['test_value']
        
        # Critical lab values that extend ICU stay
        if test_name == 'Lactate' and test_value > 4.0:
            lab_multiplier += 0.4
        elif test_name == 'Lactate' and test_value > 2.0:
            lab_multiplier += 0.2
        
        elif test_name == 'Creatinine' and test_value > 3.0:
            lab_multiplier += 0.3  # Kidney dysfunction
        elif test_name == 'Creatinine' and test_value > 1.5:
            lab_multiplier += 0.15
        
        elif test_name == 'Total Bilirubin' and test_value > 3.0:
            lab_multiplier += 0.25  # Liver dysfunction
        
        elif test_name == 'pH' and (test_value < 7.25 or test_value > 7.55):
            lab_multiplier += 0.3  # Severe acid-base imbalance
        
        elif test_name == 'Hemoglobin' and test_value < 8.0:
            lab_multiplier += 0.2  # Severe anemia
    
    return min(lab_multiplier, 1.0)  # Cap lab multiplier

def predict_length_of_stay(patient_data: Dict, vitals: Dict, lab_results: pd.DataFrame = None, current_day: int = 1) -> Dict:
    """
    Predict total length of stay in ICU
    Returns dictionary with prediction details
    """
    # Calculate base score
    base_score = calculate_base_los_score(patient_data)
    
    # Calculate severity multiplier
    severity_mult = calculate_severity_multiplier(vitals, lab_results)
    
    # Calculate predicted total LOS
    predicted_total_los = int(base_score * severity_mult)
    
    # Calculate remaining days
    remaining_days = max(0, predicted_total_los - current_day)
    
    # Calculate confidence based on data availability
    confidence = 0.65  # Base confidence
    if lab_results is not None and not lab_results.empty:
        confidence += 0.15
    if current_day > 2:  # More data after few days
        confidence += 0.1
    
    confidence = min(confidence, 0.9)
    
    # Determine discharge probability by day
    discharge_probabilities = calculate_discharge_probabilities(predicted_total_los, current_day)
    
    # Risk factors for extended stay
    risk_factors = identify_los_risk_factors(patient_data, vitals, lab_results)
    
    return {
        'predicted_total_los': predicted_total_los,
        'remaining_days': remaining_days,
        'current_day': current_day,
        'base_score': base_score,
        'severity_multiplier': severity_mult,
        'confidence': confidence,
        'discharge_probabilities': discharge_probabilities,
        'risk_factors': risk_factors
    }

def calculate_discharge_probabilities(predicted_los: int, current_day: int) -> Dict:
    """Calculate probability of discharge for next 7 days"""
    probabilities = {}
    
    for day in range(current_day + 1, current_day + 8):
        if day <= predicted_los - 2:
            prob = 0.05  # Very low if before predicted discharge
        elif day == predicted_los - 1:
            prob = 0.20
        elif day == predicted_los:
            prob = 0.50
        elif day == predicted_los + 1:
            prob = 0.30
        elif day == predicted_los + 2:
            prob = 0.15
        else:
            prob = max(0.05, 0.15 - (day - predicted_los) * 0.02)
        
        probabilities[f'day_{day}'] = prob
    
    return probabilities

def identify_los_risk_factors(patient_data: Dict, vitals: Dict, lab_results: pd.DataFrame = None) -> List[str]:
    """Identify factors that may extend length of stay"""
    risk_factors = []
    
    # Age factor
    age = patient_data.get('age', 50)
    if age > 75:
        risk_factors.append(f"Advanced age ({age} years)")
    
    # Diagnosis factors
    diagnosis = patient_data.get('diagnosis', '').lower()
    high_risk_diagnoses = ['septic shock', 'multi-organ failure', 'cardiac arrest']
    for condition in high_risk_diagnoses:
        if condition in diagnosis:
            risk_factors.append(f"High-risk diagnosis: {condition}")
    
    # Vital sign factors
    temp = vitals.get('temperature', 37.0)
    if temp > 39.0:
        risk_factors.append(f"High fever ({temp}°C)")
    elif temp < 35.0:
        risk_factors.append(f"Hypothermia ({temp}°C)")
    
    sbp = vitals.get('blood_pressure_systolic', 120)
    if sbp < 90:
        risk_factors.append(f"Hypotension ({sbp} mmHg)")
    
    spo2 = vitals.get('oxygen_saturation', 98)
    if spo2 < 90:
        risk_factors.append(f"Severe hypoxemia ({spo2}%)")
    
    # Lab-based risk factors
    if lab_results is not None and not lab_results.empty:
        for _, lab in lab_results.iterrows():
            test_name = lab['test_name']
            test_value = lab['test_value']
            
            if test_name == 'Lactate' and test_value > 4.0:
                risk_factors.append("Severe lactic acidosis")
            elif test_name == 'Creatinine' and test_value > 3.0:
                risk_factors.append("Severe kidney dysfunction")
            elif test_name == 'pH' and test_value < 7.25:
                risk_factors.append("Severe acidosis")
    
    return risk_factors

def generate_discharge_readiness_score(patient_data: Dict, vitals: Dict, lab_results: pd.DataFrame = None) -> Dict:
    """
    Generate discharge readiness score (0-100)
    Higher score = more ready for discharge
    """
    score = 50  # Start at neutral
    
    # Vital signs stability
    temp = vitals.get('temperature', 37.0)
    if 36.5 <= temp <= 37.5:
        score += 15
    elif 36.0 <= temp <= 38.0:
        score += 5
    else:
        score -= 10
    
    # Hemodynamic stability
    sbp = vitals.get('blood_pressure_systolic', 120)
    hr = vitals.get('heart_rate', 80)
    
    if 100 <= sbp <= 140 and 60 <= hr <= 90:
        score += 20
    elif 90 <= sbp <= 160 and 50 <= hr <= 100:
        score += 10
    else:
        score -= 15
    
    # Respiratory status
    spo2 = vitals.get('oxygen_saturation', 98)
    rr = vitals.get('respiratory_rate', 16)
    
    if spo2 >= 95 and 12 <= rr <= 20:
        score += 15
    elif spo2 >= 92 and 10 <= rr <= 24:
        score += 5
    else:
        score -= 10
    
    # Lab results (if available)
    if lab_results is not None and not lab_results.empty:
        lab_score = calculate_lab_discharge_score(lab_results)
        score += lab_score
    
    # Cap score between 0 and 100
    score = max(0, min(100, score))
    
    # Determine readiness level
    if score >= 80:
        readiness = "High - Consider discharge planning"
    elif score >= 60:
        readiness = "Moderate - Monitor closely"
    elif score >= 40:
        readiness = "Low - Requires continued intensive care"
    else:
        readiness = "Very Low - Critical condition"
    
    return {
        'discharge_score': score,
        'readiness_level': readiness,
        'score_components': {
            'vital_signs': min(score, 40),
            'lab_results': min(20, score - 40) if score > 40 else 0
        }
    }

def calculate_lab_discharge_score(lab_results: pd.DataFrame) -> int:
    """Calculate discharge readiness score from lab results"""
    lab_score = 0
    
    for _, lab in lab_results.iterrows():
        test_name = lab['test_name']
        test_value = lab['test_value']
        
        # Positive indicators for discharge
        if test_name == 'Lactate' and test_value <= 2.0:
            lab_score += 5
        elif test_name == 'pH' and 7.35 <= test_value <= 7.45:
            lab_score += 5
        elif test_name == 'Creatinine' and test_value <= 1.5:
            lab_score += 3
        elif test_name == 'White Blood Cells' and 4.0 <= test_value <= 11.0:
            lab_score += 3
        
        # Negative indicators
        elif test_name == 'Lactate' and test_value > 4.0:
            lab_score -= 10
        elif test_name == 'pH' and (test_value < 7.25 or test_value > 7.55):
            lab_score -= 8
        elif test_name == 'Creatinine' and test_value > 3.0:
            lab_score -= 8
    
    return max(-20, min(20, lab_score))  # Cap between -20 and +20
