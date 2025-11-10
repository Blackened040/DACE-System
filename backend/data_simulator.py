import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class EnergyDataSimulator:
    def __init__(self):
        self.base_consumption = 2.5  # kW base
        self.anomaly_probability = 0.05  # 5% de probabilidad de anomalía
        
    def generate_normal_pattern(self, hours=720):  # 30 días
        """Genera patrón normal de consumo"""
        dates = []
        consumption = []
        
        base_time = datetime.now() - timedelta(hours=hours)
        
        for i in range(hours):
            current_time = base_time + timedelta(hours=i)
            hour = current_time.hour
            
            # Patrón diario típico
            if 0 <= hour < 6:  # Madrugada
                base = self.base_consumption * 0.3
            elif 6 <= hour < 12:  # Mañana
                base = self.base_consumption * 0.8
            elif 12 <= hour < 18:  # Tarde
                base = self.base_consumption * 1.2
            else:  # Noche
                base = self.base_consumption * 1.5
                
            # Variación aleatoria normal
            variation = np.random.normal(0, 0.2)
            current_consumption = max(0, base + variation)
            
            dates.append(current_time)
            consumption.append(current_consumption)
            
        return pd.DataFrame({
            'timestamp': dates,
            'consumption_kw': consumption,
            'is_anomaly': False
        })
    
    def add_anomalies(self, df):
        """Añade anomalías al dataset"""
        anomalies_idx = np.random.choice(
            df.index, 
            size=int(len(df) * self.anomaly_probability),
            replace=False
        )
        
        for idx in anomalies_idx:
            # Tipos de anomalías
            anomaly_type = random.choice(['spike', 'drop', 'zero'])
            
            if anomaly_type == 'spike':
                df.loc[idx, 'consumption_kw'] *= random.uniform(3, 8)
            elif anomaly_type == 'drop':
                df.loc[idx, 'consumption_kw'] *= random.uniform(0.1, 0.3)
            else:  # zero
                df.loc[idx, 'consumption_kw'] = 0
                
            df.loc[idx, 'is_anomaly'] = True
            
        return df
    
    def generate_dataset(self, hours=720):
        """Genera dataset completo con anomalías"""
        normal_data = self.generate_normal_pattern(hours)
        full_data = self.add_anomalies(normal_data)
        return full_data

# Ejemplo de uso
if __name__ == "__main__":
    simulator = EnergyDataSimulator()
    dataset = simulator.generate_dataset(168)  # 1 semana
    print(f"Dataset generado: {len(dataset)} registros")
    print(f"Anomalías detectadas: {dataset['is_anomaly'].sum()}")
    dataset.to_csv('datasets/energy_consumption.csv', index=False)