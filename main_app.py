import streamlit as st
from database_functions import test_database_connection
from bed_layout_functions import render_icu_bed_layout, get_bed_status_legend
from clinical_report_functions import render_current_patient_report, render_predictive_patient_report

# Page configuration
st.set_page_config(
    page_title="GAYA-ICU Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .back-button {
        background: #007bff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin-bottom: 20px;
    }
    
    .report-type-button {
        background: #28a745;
        color: white;
        padding: 15px 30px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        margin: 10px;
        cursor: pointer;
    }
    
    .report-type-button:hover {
        background: #218838;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    load_custom_css()
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'bed_layout'
    if 'selected_patient_id' not in st.session_state:
        st.session_state.selected_patient_id = None
    if 'report_type' not in st.session_state:
        st.session_state.report_type = None
    
    # Test database connection on startup
    if 'db_tested' not in st.session_state:
        success, message = test_database_connection()
        if success:
            st.success(f"✅ Conectado ao banco de dados: {message}")
        else:
            st.error(f"❌ Erro de conexão: {message}")
        st.session_state.db_tested = True
    
    # Navigation logic
    if st.session_state.current_page == 'bed_layout':
        render_bed_layout_page()
    elif st.session_state.current_page == 'patient_report':
        render_patient_report_page()

def render_bed_layout_page():
    """Render the main bed layout page"""
    render_icu_bed_layout()
    
    # Add legend
    st.markdown(get_bed_status_legend(), unsafe_allow_html=True)

def render_patient_report_page():
    """Render patient report with type selection"""
    
    # Back button
    if st.button("← Voltar ao Layout dos Leitos", key="back_to_layout"):
        st.session_state.current_page = 'bed_layout'
        st.session_state.report_type = None
        st.rerun()
    
    patient_id = st.session_state.get('selected_patient_id')
    
    if not patient_id:
        st.error("Nenhum paciente selecionado. Retorne ao layout dos leitos.")
        return
    
    # Report type selection
    if not st.session_state.get('report_type'):
        st.markdown("""
        <div class="main-header">
            <h2>Selecione o Tipo de Relatório</h2>
            <p>Escolha entre o relatório atual do paciente ou análise preditiva</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; margin: 10px;">
                <h3>📋 Relatório Atual</h3>
                <p>Perfil completo do paciente com dados atuais, sinais vitais, resultados de exames e histórico clínico</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Acessar Relatório Atual", key="current_report", use_container_width=True):
                st.session_state.report_type = 'current'
                st.rerun()
        
        with col2:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; margin: 10px;">
                <h3>🔮 Relatório Preditivo</h3>
                <p>Análise estatística com predições de sepse, tempo de internação e recomendações baseadas em IA</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Acessar Relatório Preditivo", key="predictive_report", use_container_width=True):
                st.session_state.report_type = 'predictive'
                st.rerun()
    
    else:
        # Show change report type button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📋 Relatório Atual", key="switch_current"):
                st.session_state.report_type = 'current'
                st.rerun()
        
        with col2:
            if st.button("🔮 Relatório Preditivo", key="switch_predictive"):
                st.session_state.report_type = 'predictive'
                st.rerun()
        
        # Render selected report
        if st.session_state.report_type == 'current':
            render_current_patient_report(patient_id)
        elif st.session_state.report_type == 'predictive':
            render_predictive_patient_report(patient_id)

if __name__ == "__main__":
    main()
