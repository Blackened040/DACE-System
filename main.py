from backend.data_simulator import EnergyDataSimulator
from backend.anomaly_detector import AnomalyDetector
from backend.generate_plots import generate_all_plots, generate_results_table
import pandas as pd
import sqlite3
import os

def main():
    print("=== SISTEMA DACE - INICIALIZACIÓN ===")
    
    # Crear directorios necesarios
    os.makedirs('docs', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    
    # 1. Generar datos
    print("1. Generando datos de consumo eléctrico...")
    simulator = EnergyDataSimulator()
    data = simulator.generate_dataset(720)  # 30 días
    
    # 2. Detectar anomalías
    print("2. Entrenando modelos de detección de anomalías...")
    detector = AnomalyDetector()
    data_with_anomalies = detector.train_models(data)
    
    # 3. Guardar en base de datos
    print("3. Guardando datos en base de datos...")
    conn = sqlite3.connect('energy_consumption.db')
    data_with_anomalies.to_sql('consumption_data', conn, 
                              if_exists='replace', index=False)
    conn.close()
    
    # 4. Evaluar modelos
    print("4. Evaluando modelos...")
    evaluation = detector.evaluate_models(data_with_anomalies)
    
    # 5. Generar gráficas
    print("5. Generando gráficas y tablas...")
    generate_all_plots()
    generate_results_table()
    
    # 6. Mostrar resumen
    print("\n=== RESUMEN EJECUCIÓN ===")
    print(f"• Registros generados: {len(data_with_anomalies)}")
    print(f"• Anomalías simuladas: {data_with_anomalies['is_anomaly'].sum()}")
    print(f"• Anomalías detectadas: {data_with_anomalies['final_anomaly'].sum()}")
    
    # Precisión del modelo combinado
    if 'combined' in evaluation and 'accuracy' in evaluation['combined']:
        combined_accuracy = evaluation['combined']['accuracy']
        print(f"• Precisión combinada: {combined_accuracy:.2%}")
    
    print("\n¡Sistema DACE ejecutado exitosamente!")
    print("Ejecuta 'python backend/app.py' para iniciar la API web")
    print("Gráficas guardadas en la carpeta 'docs'")

if __name__ == '__main__':
    main()