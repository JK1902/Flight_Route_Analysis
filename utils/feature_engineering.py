import pandas as pd
from datetime import datetime

def add_features(df):
    # Map quarters to months
    df['Month'] = df['quarter'].map({1:3, 2:6, 3:9, 4:12})
    
    # Days until departure (assuming 'Year' and 'quarter' as proxy)
    df['Departure_Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-15')
    df['Days_Until_Departure'] = (df['Departure_Date'] - pd.Timestamp.today()).dt.days
    
    # Route features
    df['Route'] = df['city1'] + '-' + df['city2']
    
    # Seasonal feature
    df['Season'] = df['Month'] % 12 // 3 + 1
    
    return df

def correlation_heatmap(df, columns):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[columns].corr(), annot=True, cmap='coolwarm', ax=ax)
    return fig
