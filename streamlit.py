import streamlit as st
import joblib
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for EXACT Flask Theme Match
st.markdown("""
<style>
    /* Hide Streamlit Default UI Elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at top left, #1e1b4b, #0b0f19 50%) !important;
        background-color: #0b0f19 !important;
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
    }
    
    /* Global inputs style */
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    div[data-baseweb="input"] > input {
        color: #f1f5f9 !important;
    }
    
    /* Glassmorphic Container/Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 2rem;
    }
    
    /* Top Custom Navbar styling */
    .custom-navbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 60px;
        background: rgba(11, 15, 25, 0.8);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 999999;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 2rem;
    }
    .custom-navbar .logo {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
    }
    .custom-navbar .logo span {
        color: #3b82f6;
    }
    
    /* Adjust main content down to account for fixed navbar */
    .block-container {
        padding-top: 5rem !important;
        max-width: 800px;
    }
    
    /* Button overrides */
    .stButton>button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb !important;
    }
</style>

<!-- Inject Global CSS overrides but remove the absolute positioned custom-navbar to avoid blocking clicks -->
""", unsafe_allow_html=True)

# Helper functions for database
def get_db_connection():
    # Connect to the sqlite db
    conn = sqlite3.connect('flood_prediction.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Users (
        UserID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        Email TEXT UNIQUE NOT NULL,
        Password TEXT NOT NULL,
        Role TEXT DEFAULT 'User'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS ML_Model (
        ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
        ModelName TEXT NOT NULL,
        Accuracy REAL,
        TrainingDate DATETIME
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS Weather_Data (
        DataID INTEGER PRIMARY KEY AUTOINCREMENT,
        UserID INTEGER,
        AnnualRainfall REAL,
        CloudVisibility REAL,
        Temperature REAL,
        Humidity REAL,
        SeasonalRainfall REAL,
        FOREIGN KEY(UserID) REFERENCES Users(UserID)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS Prediction_Result (
        PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
        DataID INTEGER,
        ModelID INTEGER,
        FloodResult TEXT,
        FloodProbability REAL,
        PredictionDate DATETIME,
        FOREIGN KEY(DataID) REFERENCES Weather_Data(DataID),
        FOREIGN KEY(ModelID) REFERENCES ML_Model(ModelID)
    )''')
    # Insert a dummy model if empty
    c.execute("SELECT COUNT(*) as count FROM ML_Model")
    if c.fetchone()['count'] == 0:
        c.execute("INSERT INTO ML_Model (ModelName, Accuracy, TrainingDate) VALUES (?, ?, ?)",
                  ('XGBoost_Streamlit', 96.55, datetime.utcnow()))
    conn.commit()
    conn.close()

# Initialize DB
init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Load Models
@st.cache_resource
def load_assets():
    model = joblib.load('floods.save')
    scaler = joblib.load('transform.save')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Session State Initialization
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Top Navigation Bar (Native Streamlit)
st.markdown("<br>", unsafe_allow_html=True)
col_logo, col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([2, 1, 1, 1, 1])

with col_logo:
    st.markdown('<div style="font-size:1.8rem;font-weight:900;color:white;">Flood<span style="color:#3b82f6;">Guard</span></div>', unsafe_allow_html=True)

if st.session_state.user_id is None:
    with col_nav2:
        if st.button("Home", key="nav_home"): 
            st.session_state.page = 'Home'
            st.rerun()
    with col_nav3:
        if st.button("Login", key="nav_login"): 
            st.session_state.page = 'Login'
            st.rerun()
    with col_nav4:
        if st.button("Register", key="nav_register"): 
            st.session_state.page = 'Register'
            st.rerun()
else:
    with col_nav2:
        if st.button("Predict", key="nav_predict"): 
            st.session_state.page = 'Predict'
            st.rerun()
    with col_nav3:
        if st.button("History", key="nav_history"): 
            st.session_state.page = 'History'
            st.rerun()
    with col_nav4:
        if st.button("Logout", key="nav_logout"):
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.session_state.page = 'Home'
            st.rerun()

st.markdown("---")

# --- PAGE ROUTING ---

if st.session_state.page == 'Home':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("Predict Floods Before They Happen")
    st.write("A machine-learning powered early warning system for predicting flood risks based on historical weather patterns. Utilizing advanced algorithms like XGBoost.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login to Predict"):
            st.session_state.page = 'Login'
            st.rerun()
    with col2:
        if st.button("Register Account"):
            st.session_state.page = 'Register'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Register':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("Create an Account")
    with st.form("register_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            conn = get_db_connection()
            c = conn.cursor()
            # Note: The flask app used pbkdf2:sha256. For simplicity in Streamlit without Werkzeug, 
            # we will just do a standard sha256. If linking to the exact same db, we should use werkzeug.
            # Importing werkzeug just to be fully compatible with the Flask DB:
            try:
                from werkzeug.security import generate_password_hash
                hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
            except ImportError:
                hashed_pw = hash_password(password)
                
            try:
                c.execute("INSERT INTO Users (Name, Email, Password, Role) VALUES (?, ?, ?, ?)", (name, email, hashed_pw, 'User'))
                conn.commit()
                st.success("Registration successful! Please login.")
            except sqlite3.IntegrityError:
                st.error("Email already exists.")
            finally:
                conn.close()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Login':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("Welcome Back")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT * FROM Users WHERE Email = ?", (email,))
            user = c.fetchone()
            conn.close()
            
            if user:
                # Check password
                valid = False
                try:
                    from werkzeug.security import check_password_hash
                    valid = check_password_hash(user['Password'], password)
                except ImportError:
                    valid = (user['Password'] == hash_password(password))
                    
                if valid:
                    st.session_state.user_id = user['UserID']
                    st.session_state.user_name = user['Name']
                    st.session_state.page = 'Predict'
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            else:
                st.error("Invalid credentials.")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Predict':
    if st.session_state.user_id is None:
        st.session_state.page = 'Login'
        st.rerun()
        
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("Flood Risk Analyzer")
    st.write("Enter the meteorological parameters below to assess the likelihood of flooding.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1500.0, step=100.0)
            temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=28.5, step=1.0)
        with col2:
            seasonal_rainfall = st.number_input("Seasonal Rainfall (mm)", min_value=0.0, value=500.0, step=50.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
            
        cloud_visibility = st.number_input("Cloud Visibility / Cover (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        
        submitted = st.form_submit_button("Analyze Risk")
        
    if submitted:
        # Scale input
        input_data = pd.DataFrame({'AnnualRainfall': [annual_rainfall], 'CloudVisibility': [cloud_visibility], 'Temperature': [temperature], 'Humidity': [humidity], 'SeasonalRainfall': [seasonal_rainfall]})
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100
        
        # Save to DB
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Insert Weather Data
        c.execute('''INSERT INTO Weather_Data (UserID, AnnualRainfall, CloudVisibility, Temperature, Humidity, SeasonalRainfall)
                     VALUES (?, ?, ?, ?, ?, ?)''', 
                  (st.session_state.user_id, annual_rainfall, cloud_visibility, temperature, humidity, seasonal_rainfall))
        data_id = c.lastrowid
        
        # 2. Get Model ID
        c.execute("SELECT ModelID FROM ML_Model LIMIT 1")
        model_row = c.fetchone()
        model_id = model_row['ModelID'] if model_row else 1
        
        # 3. Insert Prediction Result
        result_text = "High Chance" if prediction == 1 else "Low Chance"
        c.execute('''INSERT INTO Prediction_Result (DataID, ModelID, FloodResult, FloodProbability, PredictionDate)
                     VALUES (?, ?, ?, ?, ?)''',
                  (data_id, model_id, result_text, probability, datetime.utcnow()))
                  
        conn.commit()
        conn.close()
        
        st.markdown("---")
        if prediction == 1:
            st.error(f"### ⚠️ High Flood Risk Detected\n**Probability: {probability:.2f}%**\nThe meteorological parameters indicate a high likelihood of flooding in the region.")
        else:
            st.success(f"### ✅ Low Flood Risk\n**Probability: {probability:.2f}%**\nThe current weather parameters suggest that the region is safe from imminent flooding.")
            
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'History':
    if st.session_state.user_id is None:
        st.session_state.page = 'Login'
        st.rerun()
        
    st.title("Your Prediction History")
    
    conn = get_db_connection()
    df_history = pd.read_sql_query(f'''
        SELECT p.PredictionDate, w.AnnualRainfall, w.SeasonalRainfall, w.Temperature, w.Humidity, w.CloudVisibility, p.FloodResult, p.FloodProbability
        FROM Prediction_Result p
        JOIN Weather_Data w ON p.DataID = w.DataID
        WHERE w.UserID = {st.session_state.user_id}
        ORDER BY p.PredictionDate DESC
    ''', conn)
    conn.close()
    
    if len(df_history) > 0:
        st.dataframe(df_history, use_container_width=True)
    else:
        st.write("You have not made any predictions yet.")
