# app.py
import streamlit as st
from streamlit_option_menu import option_menu
import importlib.util
import os

st.set_page_config(layout="wide")

# --- Sidebar navigation ---
with st.sidebar:
    selected = option_menu(
        menu_title="Flight Price Prediction Dashboard",
        options=[
            "Dataset & Cleaning",
            "Feature Engineering",
            "Model Training",
            "Prediction & Route Analysis",
            "Time Series Analysis"
        ],
        icons=["database", "gear", "bar-chart-line", "map", "clock"],
        menu_icon="cast",
        default_index=0
    )

# --- Function to load a page dynamically ---
def load_page(page_filename):
    page_path = os.path.join("pages", page_filename)
    spec = importlib.util.spec_from_file_location("page", page_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

# --- Map selected menu to page files ---
page_files = {
    "Dataset & Cleaning": "1_dataset_cleaning.py",
    "Feature Engineering": "2_feature_engineering.py",
    "Model Training": "3_model_training.py",
    "Prediction & Route Analysis": "4_prediction_analysis.py",
    "Time Series Analysis": "5_time_series_analysis.py"
}

# Load the selected page
load_page(page_files[selected])
