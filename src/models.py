from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
