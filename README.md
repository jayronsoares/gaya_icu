# GAYA-ICU Dashboard

A smart ICU monitoring system that predicts sepsis risk and patient outcomes using machine learning.

## What it does

- **Visual bed layout**: Shows 10 ICU beds with color-coded patient status
- **Sepsis prediction**: Calculates sepsis risk using vital signs and lab results  
- **Length of stay**: Predicts how long patients will need ICU care
- **Clinical reports**: Generates detailed patient reports for doctors

## How it works

The system analyzes patient data in real-time:

- **Green beds**: Stable patients (low risk)
- **Yellow beds**: Patients needing attention (moderate risk) 
- **Red beds**: Critical patients (high sepsis risk)

Click any bed to see detailed patient reports with predictions and recommendations.

## Technology

- **Frontend**: Streamlit (Python web app)
- **Database**: Supabase (PostgreSQL)
- **ML Models**: SIRS criteria + statistical analysis
- **Deployment**: Streamlit Cloud

## Setup

1. **Clone repository**
```bash
git clone https://github.com/your-username/gaya_icu.git
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Supabase**
   - Create account at supabase.com
   - Run the SQL schema to create tables
   - Add connection details to Streamlit secrets

4. **Deploy**
   - Push to GitHub
   - Connect to Streamlit Cloud
   - Add database secrets

## Usage

1. View ICU bed layout with real-time status
2. Click any bed number to access patient details
3. Choose between current report or predictive analysis
4. Monitor critical alerts for immediate attention

## Sample Data

Includes 10 sample patients with realistic medical data for demonstration purposes.

## Features

- Real-time sepsis risk calculation
- Length of stay predictions
- Discharge readiness scoring
- Clinical timeline tracking
- Vital signs trend analysis
- Lab results interpretation

## Live Demo

Visit: [gayaicu.streamlit.app](https://gayaicu.streamlit.app)

## Note

This is a prototype for demonstration purposes. Never use with real patient data without proper HIPAA compliance and medical validation.
