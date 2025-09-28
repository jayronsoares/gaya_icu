import psycopg2
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional

@st.cache_resource
def create_connection():
    """Create Supabase PostgreSQL database connection"""
    try:
        connection_string = f"postgresql://{st.secrets['supabase']['user']}:{st.secrets['supabase']['password']}@{st.secrets['supabase']['host']}:{st.secrets['supabase']['port']}/{st.secrets['supabase']['database']}?sslmode=require"
        connection = psycopg2.connect(
            host=st.secrets['supabase']['host'],
            database=st.secrets['supabase']['database'],
            user=st.secrets['supabase']['user'],
            password=st.secrets['supabase']['password'],
            port=st.secrets['supabase']['port'],
            sslmode='require'
        )
        return connection
    except psycopg2.Error as e:
        st.error(f"Supabase connection error: {e}")
        return None

@st.cache_data(ttl=30)
def fetch_all_patients() -> pd.DataFrame:
    """Fetch all patients with their current status and vital signs"""
    conn = create_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
        p.patient_id,
        p.patient_name,
        p.age,
        p.gender,
        p.admission_date,
        p.diagnosis,
        p.bed_number,
        ps.status_type,
        ps.sepsis_risk_score,
        ps.length_of_stay_prediction,
        ps.last_updated,
        ps.notes,
        vs.heart_rate,
        vs.blood_pressure_systolic,
        vs.blood_pressure_diastolic,
        vs.temperature,
        vs.respiratory_rate,
        vs.oxygen_saturation,
        vs.recorded_at as vitals_recorded_at
    FROM patients p
    LEFT JOIN patient_status ps ON p.patient_id = ps.patient_id
    LEFT JOIN vital_signs vs ON p.patient_id = vs.patient_id
    WHERE vs.recorded_at = (
        SELECT MAX(recorded_at) 
        FROM vital_signs vs2 
        WHERE vs2.patient_id = p.patient_id
    )
    ORDER BY p.bed_number
    """
    
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching patients: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_patient_details(patient_id: int) -> Optional[pd.Series]:
    """Fetch detailed information for specific patient"""
    conn = create_connection()
    if not conn:
        return None
    
    query = """
    SELECT 
        p.*,
        ps.status_type,
        ps.sepsis_risk_score,
        ps.length_of_stay_prediction,
        ps.last_updated as status_updated,
        ps.notes
    FROM patients p
    LEFT JOIN patient_status ps ON p.patient_id = ps.patient_id
    WHERE p.patient_id = %(patient_id)s
    """
    
    try:
        df = pd.read_sql(query, conn, params={"patient_id": patient_id})
        conn.close()
        return df.iloc[0] if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching patient details: {e}")
        if conn:
            conn.close()
        return None

@st.cache_data(ttl=60)
def fetch_patient_vitals_history(patient_id: int, hours: int = 48) -> pd.DataFrame:
    """Fetch vital signs history for patient"""
    conn = create_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT * FROM vital_signs 
    WHERE patient_id = %(patient_id)s 
    ORDER BY recorded_at DESC 
    LIMIT %(hours)s
    """
    
    try:
        df = pd.read_sql(query, conn, params={"patient_id": patient_id, "hours": hours})
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching vital signs: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_patient_lab_results(patient_id: int) -> pd.DataFrame:
    """Fetch lab results for patient"""
    conn = create_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
        test_name,
        test_value,
        normal_range,
        unit,
        test_date,
        CASE 
            WHEN test_name = 'White Blood Cells' AND (test_value < 4.0 OR test_value > 11.0) THEN 'abnormal'
            WHEN test_name = 'C-Reactive Protein' AND test_value > 3.0 THEN 'abnormal'
            WHEN test_name = 'Lactate' AND test_value > 2.2 THEN 'abnormal'
            WHEN test_name = 'Procalcitonin' AND test_value > 0.25 THEN 'abnormal'
            ELSE 'normal'
        END as status
    FROM lab_results 
    WHERE patient_id = %(patient_id)s 
    ORDER BY test_date DESC
    """
    
    try:
        df = pd.read_sql(query, conn, params={"patient_id": patient_id})
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching lab results: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def test_database_connection() -> Tuple[bool, str]:
    """Test database connection"""
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return True, "Supabase connection successful"
        else:
            return False, "Failed to establish connection"
    except Exception as e:
        return False, f"Connection error: {str(e)}"
