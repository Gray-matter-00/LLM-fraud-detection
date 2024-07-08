import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna()
    X = df.drop(columns=['fraud_label', 'description'])
    y = df['fraud_label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
