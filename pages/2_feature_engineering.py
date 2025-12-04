import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.title("Feature Engineering & EDA")

if 'df_clean' in st.session_state:
    df = st.session_state['df_clean'].copy()

    st.subheader("Feature Engineering")

    # --- Approximate Departure_Date from Year & Quarter ---
    def approx_departure_date(row):
        if row['quarter'] == 1:
            return pd.Timestamp(f"{row['Year']}-02-15")
        elif row['quarter'] == 2:
            return pd.Timestamp(f"{row['Year']}-05-15")
        elif row['quarter'] == 3:
            return pd.Timestamp(f"{row['Year']}-08-15")
        elif row['quarter'] == 4:
            return pd.Timestamp(f"{row['Year']}-11-15")
        else:
            return pd.NaT

    df['Departure_Date'] = df.apply(approx_departure_date, axis=1)

    # --- If Booking_Date doesn't exist, create placeholder ---
    if 'Booking_Date' not in df.columns:
        df['Booking_Date'] = df['Departure_Date'] - pd.Timedelta(days=30)  # default 30 days before

    # --- Temporal features ---
    df['Days_Until_Departure'] = (df['Departure_Date'] - df['Booking_Date']).dt.days
    df['Day'] = df['Departure_Date'].dt.day
    df['Month'] = df['Departure_Date'].dt.month
    df['Weekday'] = df['Departure_Date'].dt.weekday
    df['Season'] = df['Month'] % 12 // 3 + 1

    # --- Route features ---
    if 'Stops' not in df.columns:
        df['Stops'] = 0  # default 0, since dataset has no stops info

    # Save features for later pages
    st.session_state['df_features'] = df

    st.success("Feature engineering completed!")

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # -------------------------------
    # Scaling Comparison
    # -------------------------------
    st.subheader("Scaling Comparison")
    features_to_scale = ['Day', 'Month', 'Weekday', 'Season', 'Stops', 'Days_Until_Departure']
    # Only keep columns that exist
    features_to_scale = [feat for feat in features_to_scale if feat in df.columns]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features_to_scale])
    df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)

    fig, axes = plt.subplots(len(features_to_scale), 2, figsize=(12, 4*len(features_to_scale)))
    for i, feat in enumerate(features_to_scale):
        sns.histplot(df[feat], bins=30, ax=axes[i,0], kde=True)
        axes[i,0].set_title(f"Raw: {feat}")
        sns.histplot(df_scaled[feat], bins=30, ax=axes[i,1], kde=True)
        axes[i,1].set_title(f"Scaled: {feat}")
    plt.tight_layout()
    st.pyplot(fig)

    st.session_state['scaler'] = scaler

else:
    st.warning("Please upload and clean the dataset in the 'Dataset & Cleaning' page first.")