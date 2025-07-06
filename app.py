import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

# ----------------------
# PAGE CONFIG
# ----------------------
st.set_page_config(
    page_title="Anomaly Detection (Manual Entry)", 
    layout="wide",  # Changed from "centered" to "wide"
    initial_sidebar_state="collapsed"
)

# ----------------------
# ENHANCED CSS STYLING
# ----------------------
def load_css():
    st.markdown("""
    <style>
        /* Import modern fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --secondary: #0891b2;
            --secondary-light: #06b6d4;
            --accent: #7c3aed;
            --accent-light: #8b5cf6;
            --danger: #dc2626;
            --danger-light: #ef4444;
            --success: #059669;
            --success-light: #10b981;
            --warning: #d97706;
            --warning-light: #f59e0b;
            --orange: #ea580c;
            --orange-light: #fb7c2a;
            --teal: #0d9488;
            --teal-light: #14b8a6;
            --purple: #9333ea;
            --purple-light: #a855f7;
            --light: #f8fafc;
            --light-blue: #f0f9ff;
            --dark: #1e293b;
            --dark-light: #334155;
            --border: #e2e8f0;
            --shadow: rgba(0, 0, 0, 0.1);
            --shadow-lg: rgba(0, 0, 0, 0.15);
        }

        /* Global Styles */
        .stApp {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0f4ff 100%);
            min-height: 100vh;
        }

        .main > div {
            padding: 2rem 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Header Styles */
        .app-header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 20px;
            box-shadow: 0 10px 30px var(--shadow-lg);
            color: white;
            position: relative;
            overflow: hidden;
            animation: slideInDown 0.8s ease-out;
        }

        .app-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }

        .app-header h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 3rem;
            margin: 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
            position: relative;
            z-index: 1;
        }

        .app-header p {
            font-size: 1.2rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }

        /* Section Cards */
        .section-card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 8px 25px var(--shadow);
            border: 1px solid var(--border);
            overflow: hidden;
            margin-bottom: 2rem;
            transition: all 0.3s ease;
            animation: fadeInUp 0.6s ease-out;
        }

        .section-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 35px var(--shadow-lg);
        }

        .section-header {
            padding: 1.5rem 2rem;
            background: linear-gradient(135deg, var(--secondary), var(--teal));
            color: white;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .section-header.transaction {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
        }

        .section-header.customer {
            background: linear-gradient(135deg, var(--accent), var(--purple));
        }

        .section-header.behavioral {
            background: linear-gradient(135deg, var(--orange), var(--warning));
        }

        .section-icon {
            font-size: 1.5rem;
            opacity: 0.9;
        }

        .section-header h2 {
            margin: 0;
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.4rem;
        }

        .section-body {
            padding: 2rem;
            background: linear-gradient(135deg, #ffffff, #fafbfc);
        }

        /* Form Elements */
        .stNumberInput, .stSelectbox, .stSlider {
            margin-bottom: 1.5rem;
        }

        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {
            border-radius: 10px !important;
            border: 2px solid var(--border) !important;
            transition: all 0.3s ease !important;
            font-family: 'Inter', sans-serif !important;
        }

        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus {
            border-color: var(--primary-light) !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        }

        .stSlider > div > div > div > div {
            background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
        }

        label {
            font-weight: 600 !important;
            color: var(--dark) !important;
            font-family: 'Poppins', sans-serif !important;
            font-size: 0.95rem !important;
        }

        /* Button Styles */
        .predict-button {
            display: flex;
            justify-content: center;
            margin: 3rem 0;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
            color: white !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 1rem 3rem !important;
            border-radius: 12px !important;
            font-size: 1.1rem !important;
            font-family: 'Poppins', sans-serif !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 6px 20px var(--shadow) !important;
            position: relative !important;
            overflow: hidden !important;
        }

        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, var(--accent), var(--primary)) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px var(--shadow-lg) !important;
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        .stButton > button:active {
            transform: translateY(0) !important;
        }

        /* Result Boxes */
        .result-container {
            margin: 3rem 0;
            animation: fadeInUp 0.8s ease-out;
        }

        .result-box {
            display: flex;
            align-items: flex-start;
            padding: 2rem;
            border-radius: 16px;
            margin: 2rem 0;
            box-shadow: 0 10px 30px var(--shadow);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            animation: slideInRight 0.6s ease-out;
        }

        .result-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
            animation: shimmer 4s infinite;
        }

        .anomaly {
            background: linear-gradient(135deg, #fef2f2, #fee2e2);
            border-left: 6px solid var(--danger);
            color: var(--danger);
        }

        .normal {
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
            border-left: 6px solid var(--success);
            color: var(--success);
        }

        .result-icon {
            font-size: 3rem;
            margin-right: 2rem;
            margin-top: 0.5rem;
            animation: pulse 2s infinite;
            filter: drop-shadow(0 2px 5px rgba(0,0,0,0.1));
        }

        .result-content {
            flex: 1;
            position: relative;
            z-index: 1;
        }

        .result-content h3 {
            margin: 0 0 1rem 0;
            font-weight: 800;
            font-size: 1.5rem;
            font-family: 'Poppins', sans-serif;
            text-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .result-content p {
            margin: 0.5rem 0;
            font-size: 1rem;
            opacity: 0.9;
            line-height: 1.6;
        }

        .recommendation {
            font-style: italic;
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(255,255,255,0.7);
            border-radius: 10px;
            border-left: 4px solid currentColor;
            backdrop-filter: blur(5px);
        }

        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }

        @keyframes shimmer {
            0% {
                transform: translateX(-100%);
            }
            100% {
                transform: translateX(100%);
            }
        }

        /* Mobile Responsiveness */
        @media (max-width: 768px) {
            .main > div {
                padding: 1rem 0.5rem;
            }

            .app-header {
                padding: 1.5rem;
                margin-bottom: 2rem;
            }

            .app-header h1 {
                font-size: 2.2rem;
            }

            .section-body {
                padding: 1.5rem;
            }

            .result-box {
                flex-direction: column;
                align-items: center;
                text-align: center;
                padding: 1.5rem;
            }

            .result-icon {
                margin: 0 0 1rem 0;
            }

            .stButton > button {
                padding: 0.8rem 2rem !important;
                font-size: 1rem !important;
            }
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--light);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, var(--accent), var(--primary));
        }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# LOAD MODEL
# ----------------------
@st.cache_resource
def load_model():
    try:
        with open("best_model_xgboost.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def create_section_header(title, icon, section_type="default"):
    st.markdown(f"""
    <div class="section-card">
        <div class="section-header {section_type}">
            <span class="section-icon">{icon}</span>
            <h2>{title}</h2>
        </div>
        <div class="section-body">
    """, unsafe_allow_html=True)

def close_section():
    st.markdown("</div></div>", unsafe_allow_html=True)

def create_result_box(is_anomaly):
    if is_anomaly:
        st.markdown("""
        <div class="result-container">
            <div class="result-box anomaly">
                <div class="result-icon">🚨</div>
                <div class="result-content">
                    <h3>Anomaly Detected</h3>
                    <p>This transaction is predicted to be <strong>ANOMALOUS</strong></p>
                    <div class="recommendation">
                        ⚠️ This transaction requires immediate review and potential investigation.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-container">
            <div class="result-box normal">
                <div class="result-icon">✅</div>
                <div class="result-content">
                    <h3>Normal Transaction</h3>
                    <p>This transaction is predicted to be <strong>NORMAL</strong></p>
                    <div class="recommendation">
                        ✅ This transaction appears normal and can be processed safely.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------
# ENCODING FUNCTION
# ----------------------
def encode_input(col_name, value, options):
    le = LabelEncoder()
    le.fit(options)
    return le.transform([value])[0]

# ----------------------
# MAIN APP
# ----------------------
# Load CSS
load_css()

# Load model
model = load_model()

# App header
st.markdown("""
<div class="app-header">
    <h1> Bank Transaction Anomaly Detection</h1>
    <p>Advanced fraud detection system powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

# Transaction Information Section
create_section_header("Transaction Information", "", "transaction")
col1, col2 = st.columns(2)

with col1:
    TransactionAmount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
    TransactionType = st.selectbox("Transaction Type", ["Debit", "Credit"])

with col2:
    Channel = st.selectbox("Channel", ["Online", "ATM", "Branch"])
    TransactionDuration = st.slider("Transaction Duration (days)", 1, 300, 7)

close_section()

# Customer Information Section  
create_section_header("Customer Information", "", "customer")
col3, col4 = st.columns(2)

with col3:
    CustomerAge = st.slider("Customer Age", 18, 90, 30)
    CustomerOccupation = st.selectbox("Customer Occupation", ["Student", "Doctor", "Engineer", "Retired"])

with col4:
    AccountBalance = st.number_input("Account Balance ($)", min_value=0.0, value=1000.0, step=100.0)
    LoginAttempts = st.slider("Login Attempts", 1, 5, 1)

close_section()

# Behavioral Metrics Section
create_section_header("Behavioral Metrics", "", "behavioral")
col5, col6 = st.columns(2)

with col5:
    AccountTransactionCount = st.slider("Account Transaction Count", 1, 10, 1)

with col6:
    SpendRatio = st.slider("Spend Ratio", 0.0, 8.0, 0.3, step=0.1)

close_section()

# ----------------------
# LABEL ENCODING
# ----------------------
TransactionType_encoded = encode_input("TransactionType", TransactionType, ["Debit", "Credit"])
Channel_encoded = encode_input("Channel", Channel, ["Online", "Mobile", "ATM", "Branch"])
CustomerOccupation_encoded = encode_input("CustomerOccupation", CustomerOccupation, ["Student", "Doctor", "Engineer", "Retired", "Unemployed"])

# ----------------------
# PREDICTION
# ----------------------
st.markdown('<div class="predict-button">', unsafe_allow_html=True)
if st.button("Predict Anomaly"):
    if model:
        input_data = pd.DataFrame([[
            TransactionAmount, TransactionType_encoded, Channel_encoded, CustomerAge, CustomerOccupation_encoded,
            TransactionDuration, LoginAttempts, AccountBalance, AccountTransactionCount,
            SpendRatio
        ]], columns=[
            'TransactionAmount', 'TransactionType', 'Channel', 'CustomerAge', 'CustomerOccupation',
            'TransactionDuration', 'LoginAttempts', 'AccountBalance', 'AccountTransactionCount',
            'SpendRatio'
        ])

        prediction = model.predict(input_data)[0]
        create_result_box(prediction == 1)
    else:
        st.error("❌ Model not loaded. Please check the file.")
st.markdown('</div>', unsafe_allow_html=True)