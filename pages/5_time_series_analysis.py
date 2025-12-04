import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Time Series Analysis")

if 'df_features' in st.session_state:
    df = st.session_state['df_features']
    df_ts = df.groupby('Departure_Date')['fare'].mean().reset_index()
    df_ts.set_index('Departure_Date', inplace=True)

    decomposition = seasonal_decompose(df_ts['fare'], model='additive', period=30)
    
    st.subheader("Trend Component")
    st.line_chart(decomposition.trend.dropna())
    st.subheader("Seasonal Component")
    st.line_chart(decomposition.seasonal.dropna())
    st.subheader("Residual Component")
    st.line_chart(decomposition.resid.dropna())
else:
    st.warning("Please complete the Feature Engineering step before performing Time Series Analysis.")