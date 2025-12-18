import xgboost as xgb
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from .features import engineer_features

def train_volume_model(df):
    print("--- Training Model A: Volume Predictor ---")
    
    # Feature Engineering
    df = engineer_features(df)
    
    # Encoding
    cat_cols = ['intersection_type','maneuver']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    
    # Save Encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoder, 'models/encoder.pkl')

    # Features for Volume
    features = ['hour_sin', 'hour_cos','day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'intersection_type','maneuver']
    
    X = df[features]
    y = df['total_traffic_volume']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    rmse = root_mean_squared_error(y_test, model.predict(X_test))
    print(f"Volume Model RMSE: {rmse:.2f}")
    
    # Save
    model.save_model('models/volume_model.json')
    
    # Dataframe with predictions
    df['predicted_volume'] = model.predict(df[features])
    
    return df