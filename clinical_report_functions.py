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
        'nome_paciente': patient_info['patient_name'],
        'data_nascimento': calculate_birth_date(patient_info['age']),
        'genero': 'Masculino' if patient_info['gender'] == 'M' else 'Feminino',
        'numero_leito': patient_info['bed_number'],
        'codigo_paciente': f"A{patient_info['patient_id']:05d}",
        'duracao_estadia': f"{days_admitted} dias",
        'medico_responsavel': "Dr. Marcelo Gomes",  # Default doctor
        'diagnostico_admissional': patient_info['diagnosis'],
        'data_admissao': admission_date.strftime('%d/%m/%Y %H:%M'),
        'notas_clinicas': patient_info.get('notes', 'Acompanhamento de rotina')
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
        'frequencia_cardiaca': calculate_mean_std(vitals_history, 'heart_rate'),
        'tensao_sistolica': calculate_mean_std(vitals_history, 'blood_pressure_systolic'),
        'tensao_diastolica': calculate_mean_std(vitals_history, 'blood_pressure_diastolic'),
        'temperatura': calculate_mean_std(vitals_history, 'temperature'),
        'frequencia_respiratoria': calculate_mean_std(vitals_history, 'respiratory_rate'),
        'saturacao_oxigenio': calculate_mean_std(vitals_history, 'oxygen_saturation')
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
        
        # Map to Portuguese names
        portuguese_names = {
            'White Blood Cells': 'leucocitos',
            'Hemoglobina': 'hemoglobina',
            'Hematócrito': 'hematocrito',
            'Glucose': 'glicemia',
            'Creatinine': 'creatina',
            'Bilirrubina total': 'bilirrubina_total',
            'pH': 'ph',
            'Lactate': 'lactato_serico'
        }
        
        portuguese_name = portuguese_names.get(test_name, test_name.lower().replace(' ', '_'))
        lab_summary[portuguese_name] = f"{test_value} {unit}".strip()
    
    return lab_summary

def generate_comorbidities_and_events(patient_id: int) -> Dict:
    """Generate comorbidities and adverse events (simulated for prototype)"""
    # In a real system, this would come from patient history
    return {
        'quadro_psiquiatrico': ['Pesadelo', 'Delírios', 'Dor', 'Tristeza'],
        'comorbidades': ['Diabetes', 'Hipertensão', 'Hipoalbuminemia', 'Insuficiência renal'],
        'eventos_adversos': [
            {'evento': 'Hipoglicemia', 'frequencia': '1(0,3%)', 'medicamento': 'Midazolam'},
            {'evento': 'Pneumonia', 'frequencia': '1(0,3%)', 'medicamento': 'Vecurônio'},
            {'evento': 'Ataxia', 'frequencia': '1(0,3%)', 'medicamento': 'Neostigmina'},
            {'evento': 'Equimose', 'frequencia': '1(0,3%)', 'medicamento': 'Atracúrio'}
        ],
        'procedimentos_invasivos': ['Ventilação mecânica'],
        'local_infeccao': ['Pulmão'],
        'incidencia_sepse': ['Sepse', 'PAVM']
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
        'time': admission_date.strftime('%d/%m/%Y %H:%M'),
        'event': f'Paciente admitido para o leito {patient_info["bed_number"]}',
        'type': 'admission'
    })
    
    # Add simulated events based on days admitted
    days_admitted = (datetime.now() - admission_date).days
    
    if days_admitted >= 3:
        timeline.append({
            'time': (admission_date + timedelta(days=3)).strftime('%d/%m/%Y %H:%M'),
            'event': 'Cirurgia',
            'type': 'procedure'
        })
    
    if days_admitted >= 1:
        timeline.append({
            'time': (datetime.now() - timedelta(hours=13)).strftime('%d/%m/%Y %H:%M'),
            'event': 'Checkup de rotina',
            'type': 'checkup'
        })
        
        timeline.append({
            'time': (datetime.now() - timedelta(hours=1.5)).strftime('%d/%m/%Y %H:%M'),
            'event': 'Teste de sangue',
            'type': 'lab'
        })
    
    timeline.append({
        'time': datetime.now().strftime('%d/%m/%Y %H:%M'),
        'event': 'Neste momento',
        'type': 'current'
    })
    
    return sorted(timeline, key=lambda x: x['time'], reverse=True)

def render_current_patient_report(patient_id: int):
    """Render current patient profile report"""
    st.header("📋 Relatório Atual do Paciente")
    
    # Generate profile data
    profile = generate_current_patient_profile(patient_id)
    
    if not profile:
        st.error("Dados do paciente não encontrados.")
        return
    
    # Patient header
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{profile['nome_paciente']}")
        st.write(f"**Data de Nascimento:** {profile['data_nascimento']}")
        st.write(f"**Gênero:** {profile['genero']}")
        st.write(f"**Número do Leito:** {profile['numero_leito']}")
    
    with col2:
        st.write(f"**Código do Paciente:** {profile['codigo_paciente']}")
        st.write(f"**Duração da Estadia:** {profile['duracao_estadia']}")
        st.write(f"**Médico Responsável:** {profile['medico_responsavel']}")
    
    st.write(f"**Diagnóstico Admissional:** {profile['diagnostico_admissional']}")
    
    # Vital signs summary
    st.markdown("---")
    st.subheader("Relatório Geral - Sinais Vitais")
    st.write("*Resultados são expressos em média ± desvio padrão*")
    
    vitals_summary = generate_vital_signs_summary(patient_id)
    
    if vitals_summary:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Frequência cardíaca (bpm):** {vitals_summary.get('frequencia_cardiaca', 'N/A')}")
            st.write(f"**Tensão sistólica (mmHg):** {vitals_summary.get('tensao_sistolica', 'N/A')}")
            st.write(f"**Tensão diastólica (mmHg):** {vitals_summary.get('tensao_diastolica', 'N/A')}")
            st.write(f"**Temperatura corpórea (°C):** {vitals_summary.get('temperatura', 'N/A')}")
        
        with col2:
            st.write(f"**Frequência respiratória (ipm):** {vitals_summary.get('frequencia_respiratoria', 'N/A')}")
            st.write(f"**Saturação de oxigênio arterial (SpO2) (%):** {vitals_summary.get('saturacao_oxigenio', 'N/A')}")
    
    # Lab results
    lab_summary = generate_lab_results_summary(patient_id)
    
    if lab_summary:
        st.write(f"**Hematócrito (%):** {lab_summary.get('hematocrito', '42 ± 4.7')}")
        st.write(f"**Hemoglobina (g/dl):** {lab_summary.get('hemoglobina', '13.7 ± 4')}")
        st.write(f"**Glicemia (mg/dl):** {lab_summary.get('glicemia', 'N/A')}")
        st.write(f"**Creatina (mg/dl):** {lab_summary.get('creatina', 'N/A')}")
        st.write(f"**pH:** {lab_summary.get('ph', 'N/A')}")
        st.write(f"**Lactato sérico (mmol/L):** {lab_summary.get('lactato_serico', 'N/A')}")
        st.write(f"**Leucócitos x 10³ / µL:** {lab_summary.get('leucocitos', 'N/A')}")
    
    # Comorbidities and events
    clinical_data = generate_comorbidities_and_events(patient_id)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quadro Psiquiátrico")
        for symptom in clinical_data['quadro_psiquiatrico']:
            st.write(f"• {symptom}")
        
        st.subheader("Evento Adverso")
        for event in clinical_data['eventos_adversos']:
            st.write(f"**{event['evento']}:** {event['frequencia']} - {event['medicamento']}")
    
    with col2:
        st.subheader("Comorbidades")
        for comorbidity in clinical_data['comorbidades']:
            st.write(f"• {comorbidity}")
        
        st.subheader("Procedimento Invasivo")
        for procedure in clinical_data['procedimentos_invasivos']:
            st.write(f"• {procedure}")
    
    # Clinical timeline
    st.markdown("---")
    st.subheader("Histórico de Eventos Clínicos")
    
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
    st.header("🔮 Relatório Preditivo - Análise Estatística")
    
    # Fetch data
    patient_info = fetch_patient_details(patient_id)
    vitals_history = fetch_patient_vitals_history(patient_id, 48)
    lab_results = fetch_patient_lab_results(patient_id)
    
    if patient_info is None:
        st.error("Dados do paciente não encontrados.")
        return
    
    # Patient summary
    st.subheader(f"Análise Preditiva: {patient_info['patient_name']}")
    
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
