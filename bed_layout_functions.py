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
    
    bed_html = f"""
    <div style="
        border: 3px solid {bed_color};
        border-radius: 12px;
        padding: 15px;
        margin: 8px;
        background: linear-gradient(135deg, #ffffff 0%, {bed_color}15 100%);
        cursor: pointer;
        transition: all 0.3s ease;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    " onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)'" 
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.1)'">
        
        <div style="text-align: center;">
            <div style="color: {bed_color}; font-size: 20px; font-weight: bold; margin-bottom: 8px;">
                {patient_row['bed_number']}
            </div>
            <div style="width: 16px; height: 16px; background: {bed_color}; border-radius: 50%; margin: 0 auto 10px auto;"></div>
        </div>
        
        <div style="text-align: center; flex-grow: 1;">
            <div style="font-weight: bold; font-size: 14px; margin-bottom: 5px; color: #333;">
                {patient_row['patient_name']}
            </div>
            <div style="font-size: 12px; margin-bottom: 3px; color: #666;">
                <strong>Idade:</strong> {patient_row['age']} | <strong>Sexo:</strong> {patient_row['gender']}
            </div>
            <div style="font-size: 11px; margin-bottom: 5px; color: #666;">
                <strong>Diagnóstico:</strong> {short_diagnosis}
            </div>
        </div>
        
        <div style="text-align: center; font-size: 11px; color: #666;">
            <div><strong>Dias:</strong> {days_admitted} | <strong>Risco:</strong> {risk_score:.0f}%</div>
            <div><strong>FC:</strong> {patient_row.get('heart_rate', 'N/A')} | <strong>SpO2:</strong> {patient_row.get('oxygen_saturation', 'N/A')}%</div>
            <div><strong>Temp:</strong> {patient_row.get('temperature', 'N/A')}°C</div>
        </div>
    </div>
    """
    
    return bed_html

def render_icu_bed_layout():
    """Render the main ICU bed layout page"""
    
    # Page header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; margin-bottom: 20px; border-radius: 10px; text-align: center;">
        <h1 style="margin: 0;">🏥 GAYA-ICU - Unidade de Terapia Intensiva</h1>
        <p style="margin: 10px 0 0 0; font-size: 16px;">Monitoramento em Tempo Real - Layout dos Leitos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Atualizar Dados"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    
    with col3:
        st.write(f"**Última atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Fetch patient data
    patients_df = fetch_all_patients()
    
    if patients_df.empty:
        st.error("⚠️ Nenhum dado de paciente disponível. Verifique a conexão com o banco de dados.")
        return
    
    # Calculate statistics
    stats = calculate_icu_statistics(patients_df)
    
    # Display statistics
    display_icu_statistics(stats)
    
    # Render bed grid
    render_bed_grid(patients_df)
    
    # Display alerts
    display_patient_alerts(patients_df)
    
    # Auto-refresh
    if auto_refresh:
        st.rerun()

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
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="text-align: center; margin-bottom: 20px; color: #333;">Estatísticas da UTI</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total de Pacientes", stats['total'])
    
    with col2:
        st.metric("Estáveis", stats['stable'], delta=None, delta_color="normal")
    
    with col3:
        st.metric("Atenção", stats['alert'], delta=None, delta_color="normal")
    
    with col4:
        st.metric("Críticos", stats['critical'], delta=None, delta_color="inverse")
    
    with col5:
        st.metric("Risco Médio", f"{stats['avg_risk']:.1f}%", delta=None)

def render_bed_grid(patients_df: pd.DataFrame):
    """Render the bed grid layout"""
    st.markdown("### Layout dos Leitos - UTI")
    st.markdown("**Clique no número do leito abaixo para acessar o relatório do paciente**")
    
    # Create bed icons HTML
    beds_html = '<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin: 20px 0;">'
    
    for _, patient in patients_df.iterrows():
        status, color, risk_score = determine_bed_status(patient)
        bed_icon = create_bed_icon_html(patient, status, color, risk_score)
        beds_html += bed_icon
    
    beds_html += '</div>'
    
    st.markdown(beds_html, unsafe_allow_html=True)
    
    # Clickable buttons for each bed (Streamlit limitation workaround)
    st.markdown("---")
    
    # Create buttons in rows of 5
    rows = [patients_df.iloc[i:i+5] for i in range(0, len(patients_df), 5)]
    
    for row in rows:
        cols = st.columns(5)
        for idx, (_, patient) in enumerate(row.iterrows()):
            if idx < len(cols):
                with cols[idx]:
                    status, color, _ = determine_bed_status(patient)
                    button_label = f"{patient['bed_number']}\n{patient['patient_name'][:12]}..."
                    
                    if st.button(button_label, key=f"bed_{patient['patient_id']}", 
                               help=f"Clique para ver relatório completo de {patient['patient_name']}"):
                        st.session_state.selected_patient_id = patient['patient_id']
                        st.session_state.current_page = 'patient_report'
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
        st.error("🚨 **ALERTAS CRÍTICOS - AÇÃO IMEDIATA NECESSÁRIA**")
        
        for patient, risk_score in critical_patients:
            with st.expander(f"🔴 {patient['bed_number']} - {patient['patient_name']} (Risco: {risk_score:.0f}%)", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Diagnóstico:** {patient['diagnosis']}")
                    st.write(f"**Sinais Vitais Atuais:**")
                    st.write(f"- FC: {patient.get('heart_rate', 'N/A')} bpm")
                    st.write(f"- PA: {patient.get('blood_pressure_systolic', 'N/A')}/{patient.get('blood_pressure_diastolic', 'N/A')} mmHg")
                    st.write(f"- Temp: {patient.get('temperature', 'N/A')} °C")
                    st.write(f"- SpO2: {patient.get('oxygen_saturation', 'N/A')} %")
                
                with col2:
                    if st.button(f"📋 Relatório Completo", key=f"critical_report_{patient['patient_id']}"):
                        st.session_state.selected_patient_id = patient['patient_id']
                        st.session_state.current_page = 'patient_report'
                        st.rerun()
    
    # Alert patients
    if alert_patients:
        st.markdown("---")
        st.warning("⚠️ **PACIENTES QUE REQUEREM ATENÇÃO**")
        
        for patient, risk_score in alert_patients:
            with st.expander(f"🟡 {patient['bed_number']} - {patient['patient_name']} (Risco: {risk_score:.0f}%)"):
                st.write(f"**Diagnóstico:** {patient['diagnosis']}")
                st.write(f"**Observações:** {patient.get('notes', 'Monitoramento contínuo recomendado')}")
                
                if st.button(f"Ver Relatório", key=f"alert_report_{patient['patient_id']}"):
                    st.session_state.selected_patient_id = patient['patient_id']
                    st.session_state.current_page = 'patient_report'
                    st.rerun()

def get_bed_status_legend():
    """Return HTML for bed status legend"""
    return """
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h4 style="margin-bottom: 10px; color: #333;">Legenda dos Status dos Leitos:</h4>
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background: #28a745; border-radius: 50%; margin-right: 8px;"></div>
                <span><strong>Verde:</strong> Estável (Risco < 30%)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background: #ffc107; border-radius: 50%; margin-right: 8px;"></div>
                <span><strong>Amarelo:</strong> Atenção (Risco 30-60%)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background: #dc3545; border-radius: 50%; margin-right: 8px;"></div>
                <span><strong>Vermelho:</strong> Crítico (Risco > 60%)</span>
            </div>
        </div>
    </div>
    """
