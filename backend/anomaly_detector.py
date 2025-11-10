import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.isolation_forest = IsolationForest(
            contamination=0.05, 
            random_state=42
        )
        self.is_trained = False
        
    def extract_features(self, df):
        """Extrae características para el modelo"""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Características rolling
        df['consumption_rolling_mean_6h'] = df['consumption_kw'].rolling(
            window=6, min_periods=1
        ).mean()
        df['consumption_rolling_std_6h'] = df['consumption_kw'].rolling(
            window=6, min_periods=1
        ).std()
        
        # Llenar NaN usando bfill
        df = df.bfill()
        
        features = [
            'consumption_kw', 'hour', 'day_of_week', 'is_weekend',
            'consumption_rolling_mean_6h', 'consumption_rolling_std_6h'
        ]
        
        return df[features]
    
    def train_models(self, df):
        """Entrena ambos modelos"""
        features = self.extract_features(df)
        X_scaled = self.scaler.fit_transform(features)
        
        # K-Means
        self.kmeans.fit(X_scaled)
        distances = np.min(self.kmeans.transform(X_scaled), axis=1)
        df['kmeans_anomaly_score'] = distances
        
        # Isolation Forest
        if_scores = self.isolation_forest.fit_predict(X_scaled)
        df['isolation_forest_anomaly'] = (if_scores == -1).astype(int)
        
        # Combinar resultados
        kmeans_threshold = df['kmeans_anomaly_score'].quantile(0.95)
        df['final_anomaly'] = (
            (df['kmeans_anomaly_score'] > kmeans_threshold) |
            (df['isolation_forest_anomaly'] == 1)
        ).astype(int)
        
        self.is_trained = True
        return df
    
    def predict_anomalies(self, df):
        """Predice anomalías en nuevos datos"""
        if not self.is_trained:
            raise Exception("Modelo no entrenado. Ejecuta train_models primero.")
            
        features = self.extract_features(df)
        X_scaled = self.scaler.transform(features)
        
        # K-Means
        distances = np.min(self.kmeans.transform(X_scaled), axis=1)
        df['kmeans_anomaly_score'] = distances
        
        # Isolation Forest
        if_scores = self.isolation_forest.predict(X_scaled)
        df['isolation_forest_anomaly'] = (if_scores == -1).astype(int)
        
        # Combinar resultados
        kmeans_threshold = df['kmeans_anomaly_score'].quantile(0.95)
        df['final_anomaly'] = (
            (df['kmeans_anomaly_score'] > kmeans_threshold) |
            (df['isolation_forest_anomaly'] == 1)
        ).astype(int)
        
        return df
    
    def evaluate_models(self, df, true_anomalies_col='is_anomaly'):
        """Evalúa el desempeño de los modelos"""
        print("=== EVALUACIÓN DE MODELOS ===")
        
        # Verificar que la columna de anomalías reales existe
        if true_anomalies_col not in df.columns:
            raise KeyError(f"La columna {true_anomalies_col} no existe en el DataFrame.")
        
        # K-Means
        kmeans_pred = (df['kmeans_anomaly_score'] > 
                      df['kmeans_anomaly_score'].quantile(0.95)).astype(int)
        
        # Isolation Forest
        if_pred = df['isolation_forest_anomaly']
        
        # Combinado
        combined_pred = df['final_anomaly']
        
        print("\n--- K-Means ---")
        print(classification_report(df[true_anomalies_col], kmeans_pred))
        
        print("\n--- Isolation Forest ---")
        print(classification_report(df[true_anomalies_col], if_pred))
        
        print("\n--- Combinado ---")
        print(classification_report(df[true_anomalies_col], combined_pred))
        
        return {
            'kmeans': classification_report(df[true_anomalies_col], kmeans_pred, output_dict=True),
            'isolation_forest': classification_report(df[true_anomalies_col], if_pred, output_dict=True),
            'combined': classification_report(df[true_anomalies_col], combined_pred, output_dict=True)
        }