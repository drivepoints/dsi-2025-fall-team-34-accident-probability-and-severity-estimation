import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from .features import engineer_features

class AccidentPredictor:
    def __init__(self):
        self.vol_model = xgb.XGBRegressor()
        self.vol_model.load_model('models/volume_model.json')
        
        self.risk_model = joblib.load('models/risk_model.pkl')
        self.encoder = joblib.load('models/encoder.pkl')
        
    def predict(self, hour, day_of_week, month, intersection_type, maneuver):
        """
        Returns: (Predicted Volume, Probability of Accident)
        """
        # Prepare Input Data
        input_data = pd.DataFrame([{
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'intersection_type': intersection_type,
            'maneuver': maneuver
        }])
        
        # Feature Engineering (Same function as training)
        df_proc = engineer_features(input_data)
        
        # Apply Encoders
        # Handle unknown categories gracefully if configured in encoder
        df_proc[['intersection_type', 'maneuver']] = self.encoder.transform(df_proc[['intersection_type', 'maneuver']])

        
        # Predict Volume (Model A)
        vol_features = ['hour_sin', 'hour_cos','day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'intersection_type','maneuver']
        
        pred_vol = self.vol_model.predict(df_proc[vol_features])[0]
        df_proc['predicted_volume'] = pred_vol
        
        # Predict Risk (Model B)
        risk_features = [
           'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos','intersection_type', 'maneuver', 'predicted_volume'
        ]
        
        # Probability of class 1
        risk_prob = self.risk_model.predict_proba(df_proc[risk_features])[0][1]
        
        return pred_vol, risk_prob