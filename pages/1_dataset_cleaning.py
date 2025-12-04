import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

st.title("Dataset & Cleaning")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['df_raw'] = df.copy()
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    st.subheader("Handling Missing Values & Duplicates")
    st.write("Missing values before:", df.isnull().sum().sum())
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    st.write("Missing values after:", df.isnull().sum().sum())
    
    st.session_state['df_clean'] = df
    
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['fare'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    # Optional: Flight Routes Map
    if {'Latitude_Origin','Longitude_Origin','Latitude_Destination','Longitude_Destination'}.issubset(df.columns):
        st.subheader("Interactive Flight Routes Map")
        m = folium.Map(location=[20,0], zoom_start=2)
        for _, row in df.iterrows():
            folium.PolyLine(
                locations=[[row['Latitude_Origin'], row['Longitude_Origin']],
                           [row['Latitude_Destination'], row['Longitude_Destination']]],
                color='blue', weight=2, opacity=0.5
            ).add_to(m)
        st_folium(m, width=800, height=500)
