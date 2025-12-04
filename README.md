# Flight Route Price Prediction and Seasonal Analysis

## Project Overview
This project focuses on predicting flight fares and analyzing seasonal patterns across routes using machine learning and data analytics. The goal is to help travelers and analysts make informed decisions about flight bookings by forecasting prices and uncovering trends.

Key features include:  
- Accurate flight fare predictions using machine learning models, including Random Forest.  
- Analysis of seasonal and temporal factors affecting pricing.  
- An interactive Streamlit dashboard for real-time visualization and route comparisons.

---

## Dataset
The dataset consists of historical US flight data from 1993 to 2024, containing 245,955 rows and 23 columns. Key attributes include:  
- **Geospatial Markers:** Origin and destination airports  
- **Economic Variables:** Flight fare and miles traveled  
- **Operational Identifiers:** Airline carrier  
- **Temporal Dimensions:** Quarter, specific date, day of week  

---

## Pipeline
The project follows a structured data science pipeline:  
1. **Exploratory Data Analysis (EDA):** Correlation analysis and seasonal pattern exploration  
2. **Data Preprocessing:** Cleaning, encoding, and scaling features  
3. **Feature Engineering:** Extracting key features such as route, stops, and timing  
4. **Model Training & Evaluation:** Testing models like Linear Regression, Decision Tree, and Random Forest (best performing)  
5. **Deployment:** Interactive Streamlit dashboard for visualization and prediction  

---

## Features
- **ML-Powered Prediction:** Forecast flight fares using historical data and advanced ML models  
- **Seasonal & Temporal Analysis:** Identify trends based on months, weekdays, and days until departure  
- **Interactive Dashboard:** Compare routes, explore seasonal trends, and get real-time fare predictions  

---

## Key Findings
- **Seasonal Trends:** Q2 (summer) has highest fares, Q4 (winter) offers the best deals  
- **Booking Sweet Spot:** 4–8 weeks before departure  
- **Price Influencers:** Distance accounts for 35% of price variation; competition reduces fares by 12–18%  
- **Route Segmentation:** Four distinct route clusters identified  
- **Market Insights:** Seasonal patterns and airline share influence fare trends  

---

## Installation
To run the project locally:  
```bash
git clone https://github.com/JK1902/Flight_Route_Analysis.git
cd Flight_Route_Analysis
pip install -r requirements.txt
streamlit run app.py
