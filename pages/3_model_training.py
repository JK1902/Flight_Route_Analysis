import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Model Training & Comparison")

if 'df_features' in st.session_state:
    df = st.session_state['df_features'].copy()

    # -------------------------------
    # Prepare Features & Target
    # -------------------------------
    target = 'fare'
    feature_candidates = ['Days_Until_Departure', 'Stops', 'nsmiles', 
                          'large_ms', 'lf_ms', 'fare_lg', 'fare_low',
                          'Day', 'Month', 'Weekday', 'Season']

    # Keep only columns that exist in df
    features = [col for col in feature_candidates if col in df.columns]

    X = df[features]
    y = df[target]

    st.write(f"Features used for training: {features}")
    st.write(f"Target: {target}")

    # -------------------------------
    # Split Data
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------------------------------
    # Train Models
    # -------------------------------
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []

    best_model_name = None
    best_mape = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        results.append([name, mae, rmse, r2, mape])

        # Save Random Forest for later prediction
        if name == 'Random Forest':
            st.session_state['model_rf'] = model

        # Determine best model by lowest MAPE
        if mape < best_mape:
            best_mape = mape
            best_model_name = name
            st.session_state['best_model'] = model

    # -------------------------------
    # Show Results Table
    # -------------------------------
    results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'R²', 'MAPE'])
    st.subheader("Model Evaluation Metrics")
    st.dataframe(results_df)

    # -------------------------------
    # Predicted vs Actual Plot for Best Model
    # -------------------------------
    best_model = st.session_state['best_model']
    y_pred_best = best_model.predict(X_test)

    st.subheader(f"Predicted vs Actual ({best_model_name})")
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test, y_pred_best, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Fare")
    ax.set_ylabel("Predicted Fare")
    ax.set_title(f"{best_model_name} Predictions")
    st.pyplot(fig)

    # -------------------------------
    # Feature Importance (if Random Forest)
    # -------------------------------
    if best_model_name == 'Random Forest':
        importances = best_model.feature_importances_
        feat_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance (Random Forest)")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=feat_importance_df, ax=ax)
        st.pyplot(fig)

else:
    st.warning("Please complete feature engineering on the 'Feature Engineering & EDA' page first.")
