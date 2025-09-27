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
            st.subheader("Análise de Risco de Sepse")
            
            # Sepsis risk gauge
            fig_sepsis = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sepsis_analysis['sepsis_probability'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidade de Sepse (%)"},
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
            
            st.write(f"**Status Atual:** {sepsis_analysis['status'].title()}")
            st.write(f"**Confiança da Predição:** {sepsis_analysis['confidence']*100:.1f}%")
            st.write(f"**Tempo Estimado para Onset:** {sepsis_analysis['predicted_onset_hours']} horas")
            st.write(f"**Score SIRS:** {sepsis_analysis['sirs_score']}/4")
            
            if sepsis_analysis['risk_factors']:
                st.write("**Fatores de Risco Identificados:**")
                for factor in sepsis_analysis['risk_factors']:
                    st.write(f"• {factor}")
        
        with col2:
            st.subheader("Predição de Tempo de Internação")
            
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
                title="Probabilidade de Permanência por Dia",
                labels={'x': 'Dia de Internação', 'y': 'Probabilidade (%)'}
            )
            st.plotly_chart(fig_los, use_container_width=True)
            
            st.write(f"**Tempo Total Predito:** {los_prediction['predicted_total_los']} dias")
            st.write(f"**Dias Restantes:** {los_prediction['remaining_days']} dias")
            st.write(f"**Confiança:** {los_prediction['confidence']*100:.1f}%")
            
            if los_prediction['risk_factors']:
                st.write("**Fatores que Podem Prolongar Internação:**")
                for factor in los_prediction['risk_factors']:
                    st.write(f"• {factor}")
        
        # Discharge readiness section
        st.markdown("---")
        st.subheader("Análise de Prontidão para Alta")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Discharge readiness gauge
            fig_discharge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=discharge_readiness['discharge_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score de Prontidão (0-100)"},
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
            st.write(f"**Nível de Prontidão:** {discharge_readiness['readiness_level']}")
            
            if discharge_readiness['discharge_score'] >= 80:
                st.success("Paciente apresenta alta probabilidade de alta médica em breve")
            elif discharge_readiness['discharge_score'] >= 60:
                st.warning("Paciente requer monitoramento próximo antes da alta")
            elif discharge_readiness['discharge_score'] >= 40:
                st.error("Paciente ainda requer cuidados intensivos")
            else:
                st.error("Paciente em condição crítica - alta não recomendada")
        
        # Statistical trends
        st.markdown("---")
        st.subheader("Análise de Tendências dos Sinais Vitais")
        
        if len(vitals_history) > 1:
            # Sort by time for trends
            vitals_sorted = vitals_history.sort_values('recorded_at')
            
            # Temperature trend
            fig_temp_trend = px.line(
                vitals_sorted, 
                x='recorded_at', 
                y='temperature',
                title="Tendência da Temperatura (últimas 48h)"
            )
            fig_temp_trend.add_hline(y=38.0, line_dash="dash", line_color="red", annotation_text="Limite Febre")
            fig_temp_trend.add_hline(y=36.0, line_dash="dash", line_color="blue", annotation_text="Limite Hipotermia")
            st.plotly_chart(fig_temp_trend, use_container_width=True)
            
            # Heart rate and blood pressure trends
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hr = px.line(
                    vitals_sorted,
                    x='recorded_at',
                    y='heart_rate', 
                    title="Frequência Cardíaca"
                )
                fig_hr.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Limite Taquicardia")
                st.plotly_chart(fig_hr, use_container_width=True)
            
            with col2:
                fig_bp = px.line(
                    vitals_sorted,
                    x='recorded_at',
                    y='blood_pressure_systolic',
                    title="Pressão Arterial Sistólica"
                )
                fig_bp.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Limite Hipotensão")
                st.plotly_chart(fig_bp, use_container_width=True)
        
        # Clinical recommendations
        st.markdown("---")
        st.subheader("Recomendações Clínicas Baseadas em IA")
        
        if sepsis_analysis['sepsis_probability'] > 60:
            st.error("**AÇÃO IMEDIATA REQUERIDA:**")
            st.write("• Ativar protocolo de sepse")
            st.write("• Coletar hemoculturas e lactato")
            st.write("• Considerar administração precoce de antibióticos")
            st.write("• Monitorização contínua de sinais vitais")
        elif sepsis_analysis['sepsis_probability'] > 30:
            st.warning("**MONITORIZAÇÃO INTENSIFICADA:**")
            st.write("• Verificar sinais vitais a cada 15-30 minutos")
            st.write("• Considerar exames laboratoriais adicionais")
            st.write("• Reavaliar em 1-2 horas")
        else:
            st.success("**MONITORIZAÇÃO DE ROTINA:**")
            st.write("• Continuar plano de cuidados atual")
            st.write("• Sinais vitais conforme protocolo padrão")
    
    else:
        st.warning("Dados de sinais vitais insuficientes para análise preditiva.")
