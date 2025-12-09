## ✈️ Flight Route Analysis

**Machine Learning • Time Series • Route Visualization • Prediction Dashboard**

This project analyzes U.S. domestic flight routes using data analytics, machine learning, and interactive geospatial visualization. It is implemented as a multi-page **Streamlit dashboard** covering data exploration, feature engineering, model training, forecasting, and interactive route mapping. 

---

### 🌟 Key Features

#### 1️⃣ Dataset Exploration (EDA)
* View dataset summary, schema, missing values.
* Visualize distributions for fare, passengers, and distance.
* Filter and export insights.

#### 2️⃣ Feature Engineering
* Automated preprocessing (handling nulls, scaling, encoding).
* Domain-informed feature creation:
    * Seasonal indicators
    * Distance buckets
    * Demand categories
    * Fare normalization
* Reusable modular utilities via `utils/`.

#### 3️⃣ Model Training & Evaluation
* Train multiple ML models:
    * **Linear Regression**
    * **Random Forest**
    * **Gradient Boosting**
* Compare performance using $RMSE$, $MAE$, and $R^2$.
* Visualize prediction errors and residuals.

#### 4️⃣ Time Series Analysis
* Monthly trend analysis.
* Seasonality and year-over-year comparison.
* Time-based forecasting (**ARIMA / Prophet**).
* Trend decomposition plots. 

[Image of a time series decomposition plot showing trend, seasonality, and residual components]


#### 5️⃣ Prediction & Ranking
* Predict flight fares or passenger volumes.
* Rank routes by:
    * Highest predicted fare
    * Most demand
    * Cheapest predicted routes
* Interactive controls + downloadable CSV.

#### 6️⃣ Route Visualization (Geospatial Map)
* **Folium**-powered USA route map.
    * Route thickness = passenger volume
    * Route color = average fare
    * City-to-city popups with route details.
* Sidebar controls:
    * Fare filter
    * Passenger filter
    * Number of routes to display
    * Opacity
    * Color scheme selection
* Dynamic plotting (Seaborn & Matplotlib):
    * Fare distribution
    * Passenger distribution
    * Fare vs. passenger scatter plot

---

### 📁 Project Structure
```bash 
Flight_Route_Analysis/
│
├── app.py                      # Main Streamlit app entry point
├── README.md
├── requirements.txt
│
├── data/
│   └── flight_data.csv         # Dataset (~63 MB)
│
├── pages/                      # Streamlit multipage files
│   ├── 1_dataset_eda.py
│   ├── 2_feature_engineering.py
│   ├── 3_model_training.py
│   ├── 4_time_series_analysis.py
│   ├── 5_prediction_ranking.py
│   └── 6_route_visualization.py
│
└── utils/                      # Reusable functions
    ├── preprocessing.py
    ├── feature_engineering.py
    └── modeling.py

```
### 🚀 How to Run This Project

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 2. Launch the App
```bash
streamlit run app.py
```
The dashboard will open automatically at: http://localhost:8501

### 🧠 Technologies Used

* Python 3.x
* Streamlit
* Pandas / NumPy
* Scikit-learn
* Folium & streamlit-folium
* Matplotlib & Seaborn
* Statsmodels / Prophet (if used for forecasting)

### 📊 Dataset Information

* The dataset includes U.S. domestic flight routes with:

  * Origin & destination cities
  * Geocoded coordinates
  * Monthly passenger counts
  * Average fares
  * Route distance
  * Time period indicators
  * Large file size: ~63 MB, loaded efficiently with caching.

### 🔮 Future Enhancements

* Add real-time API for live airfare updates.
* Integrate LSTM or Prophet models for more accurate forecasting.
* Add clustering to identify route demand groups.
* Build airline-specific dashboards.
* Add performance benchmarking for models.

### 📝 License
This project is open-source under the MIT License.