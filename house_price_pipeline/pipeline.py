import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Example: load dataset (user to replace with real CSV)
# df = pd.read_csv('train.csv')
# X = df.drop('SalePrice', axis=1); y = df['SalePrice']
# For brevity, skip feature engineering — show pipeline structure

def train_pipeline(X, y):
    pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, n_jobs=-1))
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'house_pipeline.joblib')
    return pipeline