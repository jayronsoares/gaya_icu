import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from database_functions import fetch_patient_details, fetch_patient_vitals_history, fetch_patient_lab_results
from sepsis_predictions import advanced_sepsis_prediction
from length_of_stay_predictions import predict_length_of_stay, generate_discharge_readiness_score

def generate_current_patient_profile(patient_id: int) -> Dict:
    """Generate current patient profile report"""
    patient_info = fetch_patient_details(patient_id)
    
    if patient_info is None:
        return {}
    
    # Calculate current metrics
    admission_date = pd.to_datetime(patient_info['admission_date'])
    days_admitted = (datetime.now() - admission_date).days
    
    # Format patient data
    profile = {
        'patient_name': patient_info['patient_name'],
        'birth_date': calculate_birth_date(patient_info['age']),
        'gender': 'Male' if patient_info['gender'] == 'M' else 'Female',
        'bed_number': patient_info['bed_number'],
        'patient_code': f"A{patient_info['patient_id']:05d}",
        'stay_duration': f"{days_admitted} days",
        'attending_physician': "Dr. Marcelo Gomes",  # Default doctor
        'admission_diagnosis': patient_info['diagnosis'],
        'admission_date': admission_date.strftime('%m/%d/%Y %H:%M'),
        'clinical_notes': patient_info.get('notes', 'Routine follow-up')
    }
    
    return profile

def calculate_birth_date(age: int) -> str:
    """Calculate approximate birth date from age"""
    birth_year = datetime.now().year - age
    return f"01/01/{birth_year}"

def generate_vital_signs_summary(patient_id: int) -> Dict:
    """Generate vital signs summary with mean and standard deviation"""
    vitals_history = fetch_patient_vitals_history(patient_id, 72)  # Last 72 hours
    
    if vitals_history.empty:
        return {}
    
    summary = {
        'heart_rate': calculate_mean_std(vitals_history, 'heart_rate'),
        'systolic_pressure': calculate_mean_std(vitals_history, 'blood_pressure_systolic'),
        'diastolic_pressure': calculate_mean_std(vitals_history, 'blood_pressure_diastolic'),
        'temperature': calculate_mean_std(vitals_history, 'temperature'),
        'respiratory_rate': calculate_mean_std(vitals_history, 'respiratory_rate'),
        'oxygen_saturation': calculate_mean_std(vitals_history, 'oxygen_saturation')
    }
    
    return summary

def calculate_mean_std(df: pd.DataFrame, column: str) -> str:
    """Calculate mean ± standard deviation for a column"""
    if column not in df.columns or df[column].isna().all():
        return "N/A"
    
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    return f"{mean_val:.1f} ± {std_val:.1f}"

def generate_lab_results_summary(patient_id: int) -> Dict:
    """Generate lab results summary"""
    lab_results = fetch_patient_lab_results(patient_id)
    
    if lab_results.empty:
        return {}
    
    # Get latest values for each test
    latest_labs = lab_results.groupby('test_name').first().reset_index()
    
    lab_summary = {}
    for _, lab in latest_labs.iterrows():
        test_name = lab['test_name']
        test_value = lab['test_value']
        unit = lab['unit']
        
        # Map to English names
        english_names = {
            'White Blood Cells': 'white_blood_cells',
            'Hemoglobina': 'hemoglobin',
            'Hematócrito': 'hematocrit',
            'Glucose': 'glucose',
            'Creatinine': 'creatinine',
            'Bilirrubina total': 'total_bilirubin',
            'pH': 'ph',
            'Lactate': 'serum_lactate'
        }
        
        english_name = english_names.get(test_name, test_name.lower().replace(' ', '_'))
        lab_summary[english_name] = f"{test_value} {unit}".strip()
    
    return lab_summary

def generate_comorbidities_and_events(patient_id: int) -> Dict:
    """Generate comorbidities and adverse events (simulated for prototype)"""
    # In a real system, this would come from patient history
    return {
        'psychiatric_status': ['Nightmare', 'Delirium', 'Pain', 'Sadness'],
        'comorbidities': ['Diabetes', 'Hypertension', 'Hypoalbuminemia', 'Renal insufficiency'],
        'adverse_events': [
            {'event': 'Hypoglycemia', 'frequency': '1(0.3%)', 'medication': 'Midazolam'},
            {'event': 'Pneumonia', 'frequency': '1(0.3%)', 'medication': 'Vecuronium'},
            {'event': 'Ataxia', 'frequency': '1(0.3%)', 'medication': 'Neostigmine'},
            {'event': 'Ecchymosis', 'frequency': '1(0.3%)', 'medication': 'Atracurium'}
        ],
        'invasive_procedures': ['Mechanical ventilation'],
        'infection_site': ['Lung'],
        'sepsis_incidence': ['Sepsis', 'PAVM']
    }

def generate_clinical_timeline(patient_id: int) -> List[Dict]:
    """Generate clinical timeline events"""
    patient_info = fetch_patient_details(patient_id)
    
    if patient_info is None:
        return []
    
    admission_date = pd.to_datetime(patient_info['admission_date'])
    timeline = []
    
    # Add admission event
    timeline.append({
        'time': admission_date.strftime('%m/%d/%Y %H:%M'),
        'event': f'Patient admitted to bed {patient_info["bed_number"]}',
        'type': 'admission'
    })
    
    # Add simulated events based on days admitted
    days_admitted = (datetime.now() - admission_date).days
    
    if days_admitted >= 3:
        timeline.append({
            'time': (admission_date + timedelta(days=3)).strftime('%m/%d/%Y %H:%M'),
            'event': 'Surgery',
            'type': 'procedure'
        })
    
    if days_admitted >= 1:
        timeline.append({
            'time': (datetime.now() - timedelta(hours=13)).strftime('%m/%d/%Y %H:%M'),
            'event': 'Routine checkup',
            'type': 'checkup'
        })
        
        timeline.append({
            'time': (datetime.now() - timedelta(hours=1.5)).strftime('%m/%d/%Y %H:%M'),
            'event': 'Blood test',
            'type': 'lab'
        })
    
    timeline.append({
        'time': datetime.now().strftime('%m/%d/%Y %H:%M'),
        'event': 'Current time',
        'type': 'current'
    })
    
    return sorted(timeline, key=lambda x: x['time'], reverse=True)

def render_current_patient_report(patient_id: int):
    """Render current patient profile report"""
    st.header("📋 Current Patient Report")
    
    # Generate profile data
    profile = generate_current_patient_profile(patient_id)
    
    if not profile:
        st.error("Patient data not found.")
        return
    
    # Patient header
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{profile['patient_name']}")
        st.write(f"**Birth Date:** {profile['birth_date']}")
        st.write(f"**Gender:** {profile['gender']}")
        st.write(f"**Bed Number:** {profile['bed_number']}")
    
    with col2:
        st.write(f"**Patient Code:** {profile['patient_code']}")
        st.write(f"**Stay Duration:** {profile['stay_duration']}")
        st.write(f"**Attending Physician:** {profile['attending_physician']}")
    
    st.write(f"**Admission Diagnosis:** {profile['admission_diagnosis']}")
    
    # Vital signs summary
    st.markdown("---")
    st.subheader("General Report - Vital Signs")
    st.write("*Results are expressed as mean ± standard deviation*")
    
    vitals_summary = generate_vital_signs_summary(patient_id)
    
    if vitals_summary:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Heart rate (bpm):** {vitals_summary.get('heart_rate', 'N/A')}")
            st.write(f"**Systolic pressure (mmHg):** {vitals_summary.get('systolic_pressure', 'N/A')}")
            st.write(f"**Diastolic pressure (mmHg):** {vitals_summary.get('diastolic_pressure', 'N/A')}")
            st.write(f"**Body temperature (°C):** {vitals_summary.get('temperature', 'N/A')}")
        
        with col2:
            st.write(f"**Respiratory rate (rpm):** {vitals_summary.get('respiratory_rate', 'N/A')}")
            st.write(f"**Arterial oxygen saturation (SpO2) (%):** {vitals_summary.get('oxygen_saturation', 'N/A')}")
    
    # Lab results
    lab_summary = generate_lab_results_summary(patient_id)
    
    if lab_summary:
        st.write(f"**Hematocrit (%):** {lab_summary.get('hematocrit', '42 ± 4.7')}")
        st.write(f"**Hemoglobin (g/dl):** {lab_summary.get('hemoglobin', '13.7 ± 4')}")
        st.write(f"**Glucose (mg/dl):** {lab_summary.get('glucose', 'N/A')}")
        st.write(f"**Creatinine (mg/dl):** {lab_summary.get('creatinine', 'N/A')}")
        st.write(f"**pH:** {lab_summary.get('ph', 'N/A')}")
        st.write(f"**Serum lactate (mmol/L):** {lab_summary.get('serum_lactate', 'N/A')}")
        st.write(f"**White blood cells x 10³ / µL:** {lab_summary.get('white_blood_cells', 'N/A')}")
    
    # Comorbidities and events
    clinical_data = generate_comorbidities_and_events(patient_id)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Psychiatric Status")
        for symptom in clinical_data['psychiatric_status']:
            st.write(f"• {symptom}")
        
        st.subheader("Adverse Event")
        for event in clinical_data['adverse_events']:
            st.write(f"**{event['event']}:** {event['frequency']} - {event['medication']}")
    
    with col2:
        st.subheader("Comorbidities")
        for comorbidity in clinical_data['comorbidities']:
            st.write(f"• {comorbidity}")
        
        st.subheader("Invasive Procedure")
        for procedure in clinical_data['invasive_procedures']:
            st.write(f"• {procedure}")
    
    # Clinical timeline
    st.markdown("---")
    st.subheader("Clinical Events History")
    
    timeline = generate_clinical_timeline(patient_id)
    for event in timeline:
        if event['type'] == 'current':
            st.success(f"🕐 **{event['event']}** - {event['time']}")
        elif event['type'] == 'admission':
            st.info(f"🏥 **{event['event']}** - {event['time']}")
        else:
            st.write(f"📅 **{event['event']}** - {event['time']}")

def render_predictive_patient_report(patient_id: int):
    """Render predictive analysis report"""
    st.header("🔮 Predictive Report - Statistical Analysis")
    
    # Fetch data
    patient_info = fetch_patient_details(patient_id)
    vitals_history = fetch_patient_vitals_history(patient_id, 48)
    lab_results = fetch_patient_lab_results(patient_id)
    
    if patient_info is None:
        st.error("Patient data not found.")
        return
    
    # Patient summary
    st.subheader(f"Predictive Analysis: {patient_info['patient_name']}")
    
    # Current vitals for prediction
    if not vitals_history.empty:
        latest_vitals = vitals_history.iloc[0]
        vitals_dict = {
            'temperature': latest_vitals['temperature'],
            'heart_rate': latest_vitals['heart_rate'],
            'respiratory_rate': latest_vitals['respiratory_rate'],
            'oxygen_saturation': latest_vitals['oxygen_saturation'],
            'blood_pressure_systolic': latest_vitals['blood_pressure_systolic']
        }
        
        # Advanced sepsis prediction
        patient_data = {
            'age': patient_info['age'],
            'diagnosis': patient_info['diagnosis']
        }
        
        sepsis_analysis = advanced_sepsis_prediction(patient_data, vitals_history, lab_results)
        
        # Length of stay prediction
        admission_date = pd.to_datetime(patient_info['admission_date'])
        current_day = (datetime.now() - admission_date).days + 1
        
        los_prediction = predict_length_of_stay(patient_data, vitals_dict, lab_results, current_day)
        
        # Discharge readiness
        discharge_readiness = generate_discharge_readiness_score(patient_data, vitals_dict, lab_results)
        
        # Display predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sepsis Risk Analysis")
            
            # Sepsis risk gauge
            fig_sepsis = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sepsis_analysis['sepsis_probability'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sepsis Probability (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            fig_sepsis.update_layout(height=300)
            st.plotly_chart(fig_sepsis, use_container_width=True)
            
            st.write(f"**Current Status:** {sepsis_analysis['status'].title()}")
            st.write(f"**Prediction Confidence:** {sepsis_analysis['confidence']*100:.1f}%")
            st.write(f"**Estimated Time to Onset:** {sepsis_analysis['predicted_onset_hours']} hours")
            st.write(f"**SIRS Score:** {sepsis_analysis['sirs_score']}/4")
            
            if sepsis_analysis['risk_factors']:
                st.write("**Identified Risk Factors:**")
                for factor in sepsis_analysis['risk_factors']:
                    st.write(f"• {factor}")
        
        with col2:
            st.subheader("Length of Stay Prediction")
            
            # LOS prediction chart
            days_range = list(range(current_day, current_day + 15))
            probabilities = []
            
            for day in days_range:
                if day <= los_prediction['predicted_total_los']:
                    prob = max(0, 100 - (day - current_day) * 10)
                else:
                    prob = max(0, 50 - (day - los_prediction['predicted_total_los']) * 15)
                probabilities.append(prob)
            
            fig_los = px.bar(
                x=days_range,
                y=probabilities,
                title="Probability of Stay by Day",
                labels={'x': 'Hospital Day', 'y': 'Probability (%)'}
            )
            st.plotly_chart(fig_los, use_container_width=True)
            
            st.write(f"**Predicted Total Time:** {los_prediction['predicted_total_los']} days")
            st.write(f"**Remaining Days:** {los_prediction['remaining_days']} days")
            st.write(f"**Confidence:** {los_prediction['confidence']*100:.1f}%")
            
            if los_prediction['risk_factors']:
                st.write("**Factors That May Prolong Hospitalization:**")
                for factor in los_prediction['risk_factors']:
                    st.write(f"• {factor}")
        
        # Discharge readiness section
        st.markdown("---")
        st.subheader("Discharge Readiness Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Discharge readiness gauge
            fig_discharge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=discharge_readiness['discharge_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Readiness Score (0-100)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "lightgreen"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                }
            ))
            fig_discharge.update_layout(height=250)
            st.plotly_chart(fig_discharge, use_container_width=True)
        
        with col2:
            st.write(f"**Readiness Level:** {discharge_readiness['readiness_level']}")
            
            if discharge_readiness['discharge_score'] >= 80:
                st.success("Patient shows high probability of medical discharge soon")
            elif discharge_readiness['discharge_score'] >= 60:
                st.warning("Patient requires close monitoring before discharge")
            elif discharge_readiness['discharge_score'] >= 40:
                st.error("Patient still requires intensive care")
            else:
                st.error("Patient in critical condition - discharge not recommended")
        
        # Statistical trends
        st.markdown("---")
        st.subheader("Vital Signs Trend Analysis")
        
        if len(vitals_history) > 1:
            # Sort by time for trends
            vitals_sorted = vitals_history.sort_values('recorded_at')
            
            # Temperature trend
            fig_temp_trend = px.line(
                vitals_sorted, 
                x='recorded_at', 
                y='temperature',
                title="Temperature Trend (last 48h)"
            )
            fig_temp_trend.add_hline(y=38.0, line_dash="dash", line_color="red", annotation_text="Fever Threshold")
            fig_temp_trend.add_hline(y=36.0, line_dash="dash", line_color="blue", annotation_text="Hypothermia Threshold")
            st.plotly_chart(fig_temp_trend, use_container_width=True)
            
            # Heart rate and blood pressure trends
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hr = px.line(
                    vitals_sorted,
                    x='recorded_at',
                    y='heart_rate', 
                    title="Heart Rate"
                )
                fig_hr.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Tachycardia Threshold")
                st.plotly_chart(fig_hr, use_container_width=True)
            
            with col2:
                fig_bp = px.line(
                    vitals_sorted,
                    x='recorded_at',
                    y='blood_pressure_systolic',
                    title="Systolic Blood Pressure"
                )
                fig_bp.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Hypotension Threshold")
                st.plotly_chart(fig_bp, use_container_width=True)
        
        # Clinical recommendations
        st.markdown("---")
        st.subheader("AI-Based Clinical Recommendations")
        
        if sepsis_analysis['sepsis_probability'] > 60:
            st.error("**IMMEDIATE ACTION REQUIRED:**")
            st.write("• Activate sepsis protocol")
            st.write("• Collect blood cultures and lactate")
            st.write("• Consider early antibiotic administration")
            st.write("• Continuous vital signs monitoring")
        elif sepsis_analysis['sepsis_probability'] > 30:
            st.warning("**INTENSIVE MONITORING:**")
            st.write("• Check vital signs every 15-30 minutes")
            st.write("• Consider additional laboratory tests")
            st.write("• Reassess in 1-2 hours")
        else:
            st.success("**ROUTINE MONITORING:**")
            st.write("• Continue current care plan")
            st.write("• Vital signs according to standard protocol")
    
    else:
        st.warning("Insufficient vital signs data for predictive analysis.")
