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
        background: linear-gradient(90deg, #87CEEB 0%, #87CEEB 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .header-container {
        background: linear-gradient(90deg, #87CEEB 0%, #87CEEB 100%);
        color: white;
        padding: 20px;
        margin-bottom: 30px;
        border-radius: 10px;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
        padding: 20px;
        background: #E6F3FF;
        border-radius: 10px;
        border: 2px solid #87CEEB;
    }
    
    .bed-icon {
        border: 3px solid;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        background: white;
        box-shadow: 0 4px 8px rgba(135, 206, 235, 0.3);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .bed-icon:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(135, 206, 235, 0.5);
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
            st.success(f"✅ Connected to database: {message}")
        else:
            st.error(f"❌ Connection error: {message}")
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
    if st.button("← Back to Bed Layout", key="back_to_layout"):
        st.session_state.current_page = 'bed_layout'
        st.session_state.report_type = None
        st.rerun()
    
    patient_id = st.session_state.get('selected_patient_id')
    
    if not patient_id:
        st.error("No patient selected. Return to bed layout.")
        return
    
    # Report type selection
    if not st.session_state.get('report_type'):
        st.markdown("""
        <div class="main-header">
            <h2><i class="fas fa-clipboard-list medical-icon"></i>Select Report Type</h2>
            <p>Choose between current patient report or predictive analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="report-card">
                <h3><i class="fas fa-file-medical medical-icon"></i>Current Report</h3>
                <p>Complete patient profile with current data, vital signs, test results and clinical history</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Access Current Report", key="current_report", use_container_width=True):
                st.session_state.report_type = 'current'
                st.rerun()
        
        with col2:
            st.markdown("""
            <div class="report-card">
                <h3><i class="fas fa-chart-line medical-icon"></i>Predictive Report</h3>
                <p>Statistical analysis with sepsis predictions, length of stay and AI-based recommendations</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Access Predictive Report", key="predictive_report", use_container_width=True):
                st.session_state.report_type = 'predictive'
                st.rerun()
    
    else:
        # Show change report type button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📋 Current Report", key="switch_current"):
                st.session_state.report_type = 'current'
                st.rerun()
        
        with col2:
            if st.button("🔮 Predictive Report", key="switch_predictive"):
                st.session_state.report_type = 'predictive'
                st.rerun()
        
        # Render selected report
        if st.session_state.report_type == 'current':
            render_current_patient_report(patient_id)
        elif st.session_state.report_type == 'predictive':
            render_predictive_patient_report(patient_id)

if __name__ == "__main__":
    main()
