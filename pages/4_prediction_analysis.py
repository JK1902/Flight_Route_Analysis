import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

st.title("Prediction & Route Analysis")

# Check if previous steps are complete
if 'df_features' not in st.session_state or 'model_rf' not in st.session_state:
    st.warning("Please complete Feature Engineering & Model Training pages first.")
else:
    df = st.session_state['df_features'].copy()
    rf_model = st.session_state['model_rf']

    # -------------------------------
    # Prediction Inputs
    # -------------------------------
    st.subheader("Predict Flight Fare")

    # Origin / Destination selection
    origin = st.selectbox("Select Origin City", df['city1'].unique())
    destination = st.selectbox("Select Destination City", df['city2'].unique())
    stops = st.number_input("Number of Stops", min_value=0, max_value=3, value=0)
    days_until_departure = st.number_input("Days Until Departure", min_value=1, max_value=365, value=30)
    nsmiles = df[(df['city1'] == origin) & (df['city2'] == destination)]['nsmiles'].mean()
    large_ms = df[(df['city1'] == origin) & (df['city2'] == destination)]['large_ms'].mean()
    lf_ms = df[(df['city1'] == origin) & (df['city2'] == destination)]['lf_ms'].mean()
    fare_lg = df[(df['city1'] == origin) & (df['city2'] == destination)]['fare_lg'].mean()
    fare_low = df[(df['city1'] == origin) & (df['city2'] == destination)]['fare_low'].mean()

    day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    weekday = st.number_input("Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
    season = (month % 12) // 3 + 1

    # Prepare input for model
    input_df = pd.DataFrame({
        'Days_Until_Departure':[days_until_departure],
        'Stops':[stops],
        'nsmiles':[nsmiles],
        'large_ms':[large_ms],
        'lf_ms':[lf_ms],
        'fare_lg':[fare_lg],
        'fare_low':[fare_low],
        'Day':[day],
        'Month':[month],
        'Weekday':[weekday],
        'Season':[season]
    })

    if st.button("Predict Fare"):
        pred_fare = rf_model.predict(input_df)[0]
        st.success(f"Predicted Fare: ${pred_fare:.2f}")

    # -------------------------------
    # Route Ranking by Predicted Fare
    # -------------------------------
    st.subheader("Route Ranking by Predicted Fare")

    sample_routes = df[['city1','city2','Days_Until_Departure','Stops','nsmiles','large_ms','lf_ms','fare_lg','fare_low','Day','Month','Weekday','Season']].drop_duplicates()
    sample_routes['Predicted_Fare'] = rf_model.predict(sample_routes[[
        'Days_Until_Departure','Stops','nsmiles','large_ms','lf_ms','fare_lg','fare_low','Day','Month','Weekday','Season'
    ]])
    route_ranking = sample_routes.sort_values('Predicted_Fare').reset_index(drop=True)
    st.dataframe(route_ranking[['city1','city2','Predicted_Fare']].head(20))

    # -------------------------------
    # Seasonal Trends
    # -------------------------------
    st.subheader("Seasonal Trends of Fares")

    seasonal_df = df.groupby('Month')['fare'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=seasonal_df, x='Month', y='fare', marker='o', ax=ax)
    ax.set_title("Average Fare per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Fare")
    st.pyplot(fig)

    # -------------------------------
    # Interactive Folium Map
    # -------------------------------
    # st.subheader("Flight Route Map")

    # # Parse coordinates safely
    # def parse_coords(coord):
    #     try:
    #         if isinstance(coord, str):
    #             coord = coord.replace("(", "").replace(")", "")
    #             lat, lon = coord.split(",")
    #             return float(lat), float(lon)
    #         elif isinstance(coord, (tuple, list)):
    #             return tuple(coord)
    #     except Exception:
    #         return (0,0)
    #     return (0,0)

    # df['Origin_coords'] = df['Geocoded_City1'].apply(parse_coords)
    # df['Dest_coords'] = df['Geocoded_City2'].apply(parse_coords)

    # # Center map
    # lat_center = df['Origin_coords'].apply(lambda x: x[0]).mean()
    # lon_center = df['Origin_coords'].apply(lambda x: x[1]).mean()
    # m = folium.Map(location=[lat_center, lon_center], zoom_start=4)

    # for _, row in df.iterrows():
    #     origin = row['Origin_coords']
    #     dest = row['Dest_coords']
    #     if origin == (0,0) or dest == (0,0):
    #         continue
    #     folium.PolyLine([origin, dest], color='blue', weight=2, opacity=0.5).add_to(m)
    #     folium.Marker(origin, tooltip=row['city1']).add_to(m)
    #     folium.Marker(dest, tooltip=row['city2']).add_to(m)

    # st_folium(m, width=800, height=600)
    
