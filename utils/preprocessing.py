import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['fare', 'nsmiles', 'city1', 'city2'])
    
    # Convert data types
    df['Year'] = df['Year'].astype(int)
    df['quarter'] = df['quarter'].astype(int)
    df['passengers'] = df['passengers'].astype(int)
    
    return df
