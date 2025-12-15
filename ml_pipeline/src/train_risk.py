import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OrdinalEncoder

def train_risk_model(df):
    print("--- Training Model B: Accident Probability ---")
    
    # df already has 'predicted_volume' and engineered features from the previous step
    
    # Create Weighted Dataset (The 0/1 Splitting Strategy)
    # Row for Accidents (Positive Class)
    df_pos = df[df['fatal_accident_count'] > 0].copy()
    df_pos['target'] = 1
    df_pos['sample_weight'] = df_pos['fatal_accident_count']
    
    # Row for Safe Passage (Negative Class)
    df_neg = df.copy()
    df_neg['target'] = 0
    # Safe cars = Total Volume - Accidents
    df_neg['sample_weight'] = df_neg['total_traffic_volume'] - df_neg['fatal_accident_count']
    # Remove rows where volume < accidents (bad data)
    df_neg = df_neg[df_neg['sample_weight'] > 0]
    
    # Combine
    df_final = pd.concat([df_pos, df_neg], axis=0)
    
    # Select Features
    features = [
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
        'intersection_type', 'maneuver', 'predicted_volume'
    ]
    
    X = df_final[features]
    y = df_final['target']
    weights = df_final['sample_weight']
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    # Train Base Model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='logloss'
    )
    base_model.fit(X_train, y_train, sample_weight=w_train)
    
    # Calibrate (To get true probabilities)
    cal_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    cal_model.fit(X_test, y_test, sample_weight=w_test)
    
    # Save
    joblib.dump(cal_model, 'models/risk_model.pkl')
    print("Risk Model Trained and Saved.")