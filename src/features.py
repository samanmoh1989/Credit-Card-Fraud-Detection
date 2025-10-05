from sklearn.preprocessing import StandardScaler

def scale_features(df):
    """Scale Time and Amount columns."""
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    return df, scaler
