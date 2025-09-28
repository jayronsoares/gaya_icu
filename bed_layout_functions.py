import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from database_functions import fetch_all_patients
from sepsis_predictions import calculate_sepsis_probability, advanced_sepsis_prediction
from length_of_stay_predictions import predict_length_of_stay

def determine_bed_status(patient_row: pd.Series) -> Tuple[str, str, float]:
    """
    Determine bed color status based on patient condition
    Returns: (status, color, risk_score)
    """
    vitals = {
        'temperature': patient_row.get('temperature', 37.0),
        'heart_rate': patient_row.get('heart_rate', 80),
        'respiratory_rate': patient_row.get('respiratory_rate', 16),
        'oxygen_saturation': patient_row.get('oxygen_saturation', 98),
        'blood_pressure_systolic': patient_row.get('blood_pressure_systolic', 120)
    }
    
    # Calculate sepsis probability
    probability, status, _ = calculate_sepsis_probability(vitals)
    
    # Determine color based on status
    color_map = {
        'stable': '#28a745',    # Green
        'alert': '#ffc107',     # Yellow  
        'critical': '#dc3545'   # Red
    }
    
    return status, color_map.get(status, '#6c757d'), probability

def create_bed_icon_html(patient_row: pd.Series, bed_status: str, bed_color: str, risk_score: float) -> str:
    """Create HTML for individual bed icon"""
    
    # Calculate days admitted
    admission_date = pd.to_datetime(patient_row['admission_date'])
    days_admitted = (datetime.now() - admission_date).days
    
    # Truncate diagnosis if too long
    diagnosis = str(patient_row.get('diagnosis', 'N/A'))
    short_diagnosis = diagnosis[:25] + '...' if len(diagnosis) > 25 else diagnosis
    
    # FIXED: Remove line breaks and format as single line HTML
    bed_html = f'<div style="border: 3px solid {bed_color}; border-radius: 12px; padding: 15px; margin: 8px; background: linear-gradient(135deg, #ffffff 0%, {bed_color}15 100%); cursor: pointer; transition: all 0.3s ease; min-height: 180px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 2px 8px rgba(0,0,0,0.1); position: relative;" onmouseover="this.style.transform=\'translateY(-3px)\'; this.style.boxShadow=\'0 4px 12px rgba(0,0,0,0.2)\';" onmouseout="this.style.transform=\'translateY(0)\'; this.style.boxShadow=\'0 2px 8px rgba(0,0,0,0.1)\';"><div style="text-align: center;"><div style="color: {bed_color}; font-size: 20px; font-weight: bold; margin-bottom: 8px;">{patient_row["bed_number"]}</div><i class="fas fa-bed" style="font-size: 24px; color: {bed_color}; margin: 0 auto 10px auto;"></i></div><div style="text-align: center; flex-grow: 1;"><div style="font-weight: bold; font-size: 14px; margin-bottom: 5px; color: #333;">{patient_row["patient_name"]}</div><div style="font-size: 12px; margin-bottom: 3px; color: #666;"><strong>Age:</strong> {patient_row["age"]} | <strong>Gender:</strong> {patient_row["gender"]}</div><div style="font-size: 11px; margin-bottom: 5px; color: #666;"><strong>Diagnosis:</strong> {short_diagnosis}</div></div><div style="text-align: center; font-size: 11px; color: #666;"><div><strong>Days:</strong> {days_admitted} | <strong>Risk:</strong> {risk_score:.0f}%</div><div><strong>HR:</strong> {patient_row.get("heart_rate", "N/A")} | <strong>SpO2:</strong> {patient_row.get("oxygen_saturation", "N/A")}%</div><div><strong>Temp:</strong> {patient_row.get("temperature", "N/A")}°C</div></div><div style="position: absolute; top: 5px; right: 5px; background: rgba(255,255,255,0.9); border-radius: 4px; padding: 4px 6px; font-size: 9px; color: #666;">ID: {patient_row["patient_id"]}</div></div>'
    
    return bed_html

def render_icu_bed_layout():
    """Render the main ICU bed layout page"""
    
    # Page header - Updated to match main app styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #B0E0E6 0%, #87CEEB 100%); color: white; padding: 25px; margin-bottom: 25px; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(135, 206, 235, 0.3);">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 600;"><i class="fas fa-hospital medical-icon"></i>GAYA-ICU - Intensive Care Unit</h1>
        <p style="margin: 12px 0 0 0; font-size: 1.1rem; opacity: 0.95;">Real-Time Monitoring - Bed Layout</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons - FIXED to prevent recursion
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Refresh Data", help="Clear cache and reload patient data", key="refresh_data_btn"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        # AUTO-REFRESH DISABLED to prevent infinite recursion
        st.write("Auto-refresh: Disabled")
        # auto_refresh = st.checkbox("Auto-refresh (30s)", value=False, help="Automatically refresh data every 30 seconds")
    
    with col3:
        st.write(f"**Last updated:** {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}")
    
    # Fetch patient data
    patients_df = fetch_all_patients()
    
    if patients_df.empty:
        st.error("⚠️ No patient data available. Check database connection.")
        return
    
    # Calculate statistics
    stats = calculate_icu_statistics(patients_df)
    
    # Display statistics
    display_icu_statistics(stats)
    
    # Render bed grid
    render_bed_grid(patients_df)
    
    # Display alerts
    display_patient_alerts(patients_df)
    
    # AUTO-REFRESH COMPLETELY REMOVED to prevent recursion

def calculate_icu_statistics(patients_df: pd.DataFrame) -> Dict:
    """Calculate ICU statistics"""
    stats = {'total': 0, 'stable': 0, 'alert': 0, 'critical': 0, 'avg_risk': 0}
    
    if not patients_df.empty:
        stats['total'] = len(patients_df)
        
        risk_scores = []
        for _, patient in patients_df.iterrows():
            status, _, risk_score = determine_bed_status(patient)
            risk_scores.append(risk_score)
            stats[status] += 1
        
        stats['avg_risk'] = sum(risk_scores) / len(risk_scores) if risk_scores else 0
    
    return stats

def display_icu_statistics(stats: Dict):
    """Display ICU statistics dashboard"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #F0F8FF 0%, #E6F3FF 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 2px solid #B0E0E6; box-shadow: 0 2px 8px rgba(135, 206, 235, 0.2);">
        <h3 style="text-align: center; margin-bottom: 20px; color: #333;"><i class="fas fa-chart-bar medical-icon"></i>ICU Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Patients", stats['total'])
    
    with col2:
        st.metric("Stable", stats['stable'], delta=None, delta_color="normal")
    
    with col3:
        st.metric("Alert", stats['alert'], delta=None, delta_color="normal")
    
    with col4:
        st.metric("Critical", stats['critical'], delta=None, delta_color="inverse")
    
    with col5:
        st.metric("Average Risk", f"{stats['avg_risk']:.1f}%", delta=None)

def render_bed_grid(patients_df: pd.DataFrame):
    """Render the bed grid layout - FIXED HTML rendering"""
    st.markdown("### <i class='fas fa-bed'></i> Bed Layout - ICU", unsafe_allow_html=True)
    st.markdown("**Click on the bed number below to access the patient report**")
    
    # FIXED: Render beds in smaller chunks to avoid HTML parsing issues
    beds_per_row = 5
    rows = [patients_df.iloc[i:i+beds_per_row] for i in range(0, len(patients_df), beds_per_row)]
    
    for row in rows:
        # Create row HTML with proper grid layout
        row_html = '<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin: 20px 0;">'
        
        for _, patient in row.iterrows():
            status, color, risk_score = determine_bed_status(patient)
            bed_icon = create_bed_icon_html(patient, status, color, risk_score)
            row_html += bed_icon
        
        row_html += '</div>'
        
        # Render each row separately to avoid large HTML string issues
        st.markdown(row_html, unsafe_allow_html=True)
    
    # Clickable buttons for each bed (Enhanced with direct report access)
    st.markdown("---")
    
    # Create buttons in rows of 5 with report type selection
    button_rows = [patients_df.iloc[i:i+5] for i in range(0, len(patients_df), 5)]
    
    for row_idx, row in enumerate(button_rows):
        cols = st.columns(5)
        for idx, (_, patient) in enumerate(row.iterrows()):
            if idx < len(cols):
                with cols[idx]:
                    status, color, _ = determine_bed_status(patient)
                    
                    # Two buttons per patient - Current and Predictive
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # FIXED: Added unique keys to prevent conflicts
                        if st.button(f"📋 {patient['bed_number']}", 
                                   key=f"current_{patient['patient_id']}_{row_idx}_{idx}", 
                                   help=f"Current report for {patient['patient_name']}",
                                   use_container_width=True):
                            st.session_state.selected_patient_id = patient['patient_id']
                            st.session_state.current_page = 'patient_report'
                            st.session_state.report_type = 'current'
                            st.rerun()
                    
                    with col_b:
                        # FIXED: Added unique keys to prevent conflicts
                        if st.button(f"🔮 {patient['bed_number']}", 
                                   key=f"predictive_{patient['patient_id']}_{row_idx}_{idx}", 
                                   help=f"Predictive analysis for {patient['patient_name']}",
                                   use_container_width=True):
                            st.session_state.selected_patient_id = patient['patient_id']
                            st.session_state.current_page = 'patient_report'
                            st.session_state.report_type = 'predictive'
                            st.rerun()

def display_patient_alerts(patients_df: pd.DataFrame):
    """Display critical and alert patients"""
    critical_patients = []
    alert_patients = []
    
    for _, patient in patients_df.iterrows():
        status, _, risk_score = determine_bed_status(patient)
        
        if status == 'critical':
            critical_patients.append((patient, risk_score))
        elif status == 'alert':
            alert_patients.append((patient, risk_score))
    
    # Critical alerts
    if critical_patients:
        st.markdown("---")
        st.error("🚨 **CRITICAL ALERTS - IMMEDIATE ACTION REQUIRED**")
        
        for patient, risk_score in critical_patients:
            with st.expander(f"🔴 {patient['bed_number']} - {patient['patient_name']} (Risk: {risk_score:.0f}%)", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Diagnosis:** {patient['diagnosis']}")
                    st.write(f"**Current Vital Signs:**")
                    st.write(f"- HR: {patient.get('heart_rate', 'N/A')} bpm")
                    st.write(f"- BP: {patient.get('blood_pressure_systolic', 'N/A')}/{patient.get('blood_pressure_diastolic', 'N/A')} mmHg")
                    st.write(f"- Temp: {patient.get('temperature', 'N/A')} °C")
                    st.write(f"- SpO2: {patient.get('oxygen_saturation', 'N/A')} %")
                
                with col2:
                    # FIXED: Added unique key to prevent conflicts
                    if st.button(f"📋 Complete Report", key=f"critical_report_{patient['patient_id']}_alert"):
                        st.session_state.selected_patient_id = patient['patient_id']
                        st.session_state.current_page = 'patient_report'
                        st.session_state.report_type = 'current'
                        st.rerun()
    
    # Alert patients
    if alert_patients:
        st.markdown("---")
        st.warning("⚠️ **PATIENTS REQUIRING ATTENTION**")
        
        for patient, risk_score in alert_patients:
            with st.expander(f"🟡 {patient['bed_number']} - {patient['patient_name']} (Risk: {risk_score:.0f}%)"):
                st.write(f"**Diagnosis:** {patient['diagnosis']}")
                st.write(f"**Notes:** {patient.get('notes', 'Continuous monitoring recommended')}")
                
                # FIXED: Added unique key to prevent conflicts
                if st.button(f"View Report", key=f"alert_report_{patient['patient_id']}_warning"):
                    st.session_state.selected_patient_id = patient['patient_id']
                    st.session_state.current_page = 'patient_report'
                    st.session_state.report_type = 'current'
                    st.rerun()

def get_bed_status_legend():
    """Return HTML for bed status legend"""
    return """
    <div style="background: linear-gradient(135deg, #F0F8FF 0%, #E6F3FF 100%); padding: 20px; border-radius: 12px; margin: 25px 0; border: 2px solid #B0E0E6; box-shadow: 0 2px 8px rgba(135, 206, 235, 0.2);">
        <h4 style="margin-bottom: 15px; color: #333; text-align: center;"><i class="fas fa-info-circle medical-icon"></i>Bed Status Legend:</h4>
        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div style="display: flex; align-items: center;">
                <i class="fas fa-bed" style="color: #28a745; font-size: 20px; margin-right: 8px;"></i>
                <span><strong>Green:</strong> Stable (Risk < 30%)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <i class="fas fa-bed" style="color: #ffc107; font-size: 20px; margin-right: 8px;"></i>
                <span><strong>Yellow:</strong> Alert (Risk 30-60%)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <i class="fas fa-bed" style="color: #dc3545; font-size: 20px; margin-right: 8px;"></i>
                <span><strong>Red:</strong> Critical (Risk > 60%)</span>
            </div>
        </div>
    </div>
    """
