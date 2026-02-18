import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from prophet import Prophet

# --- Page config ---
st.set_page_config(page_title="Electricity Forecast Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Theme state ---
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Light'

# --- Apply glassmorphic teal theme ---
def apply_theme(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {background: rgba(10,50,70,0.7); backdrop-filter: blur(20px); color: #fff;}
        [data-testid="stSidebar"] {background: rgba(0,70,100,0.6); backdrop-filter: blur(20px);}
        .css-1d391kg {background: rgba(0,150,180,0.3); backdrop-filter: blur(15px); border-radius: 15px;}
        h1, h2, h3 {color: #b2f7ff;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {background: rgba(0,150,180,0.4); backdrop-filter: blur(20px);}
        [data-testid="stSidebar"] {background: rgba(0,100,150,0.4); backdrop-filter: blur(20px);}
        .css-1d391kg {background: rgba(0,200,220,0.3); backdrop-filter: blur(15px); border-radius: 15px;}
        h1, h2, h3 {color: #ffffff;}
        </style>
        """, unsafe_allow_html=True)

apply_theme(st.session_state['theme'])

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Select Page", ["Home", "EDA", "Model Testing", "Settings"])

# -----------------------------
# HOME: Upload + Table
# -----------------------------
if menu == "Home":
    st.title("⚡ Electricity Forecast Dashboard")
    st.subheader("Upload your dataset")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        # Store dataframe in session_state
        st.session_state['df'] = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        st.success("Dataset loaded successfully!")
    
    if 'df' in st.session_state:
        st.subheader("Editable Dataset Table")
        edited_df = st.data_editor(st.session_state['df'], num_rows="dynamic", use_container_width=True)
        if st.button("Reset Table"):
            edited_df = st.session_state['df'].copy()
            st.success("Table reset to original dataset.")

# -----------------------------
# EDA Tab
# -----------------------------
elif menu == "EDA":
    st.header("Exploratory Data Analysis")
    if 'df' not in st.session_state:
        st.warning("Please upload dataset first on Home tab.")
    else:
        df = st.session_state['df']
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.dataframe(df.head())
        st.write("Missing values:")
        st.dataframe(df.isna().sum())
        
        st.subheader("Sample Plots")
        sample_clients = df.columns[:3]
        plt.figure(figsize=(12,5))
        for client in sample_clients:
            plt.plot(df[client], label=client)
        plt.title("Sample Electricity Consumption")
        plt.xlabel("Date")
        plt.ylabel("Consumption")
        plt.legend()
        st.pyplot(plt)

# -----------------------------
# Model Testing Tab
# -----------------------------
elif menu == "Model Testing":
    st.header("ARIMA & Prophet Forecasting")
    if 'df' not in st.session_state:
        st.warning("Please upload dataset first on Home tab.")
    else:
        df = st.session_state['df']
        client = st.selectbox("Select Client:", df.columns.tolist())
        model_choice = st.radio("Select Model:", ["ARIMA", "Prophet"])
        horizon = st.slider("Forecast horizon (days):", 7, 90, 30)
        
        if st.button("Run Forecast"):
            series = df[client].fillna(method='ffill').fillna(method='bfill')
            train = series[:-horizon]
            test = series[-horizon:]
            
            if model_choice == "ARIMA":
                model_arima = sm.tsa.ARIMA(train, order=(1,1,1))
                arima_result = model_arima.fit()
                forecast = arima_result.forecast(steps=horizon)
            else:
                prophet_df = series.reset_index()
                prophet_df.columns = ['ds','y']
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
                prophet_df = prophet_df.dropna()
                model_prophet = Prophet(daily_seasonality=True)
                model_prophet.fit(prophet_df[:-horizon])
                future = model_prophet.make_future_dataframe(periods=horizon, freq='D')
                forecast = model_prophet.predict(future)['yhat'][-horizon:].values
            
            # Metrics
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mape = np.mean(np.abs((test-forecast)/test))*100
            
            st.subheader("Forecast Metrics")
            st.write(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
            
            # Plot
            plt.figure(figsize=(12,5))
            plt.plot(series[-horizon*2:], label="Actual")
            plt.plot(test.index, forecast, label="Forecast", color='red')
            plt.title(f"{model_choice} Forecast - {client}")
            plt.xlabel("Date")
            plt.ylabel("Electricity Consumption")
            plt.legend()
            st.pyplot(plt)

# -----------------------------
# Settings Tab
# -----------------------------
elif menu == "Settings":
    st.header("Settings")
    theme_option = st.radio("Select Theme:", ["Light", "Dark"])
    if st.session_state['theme'] != theme_option:
        st.session_state['theme'] = theme_option
        apply_theme(theme_option)
        st.success(f"Theme changed to {theme_option}!")








