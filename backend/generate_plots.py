import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3


# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_all_plots():
    """Genera todas las gráficas para el proyecto"""
    # Conectar a la base de datos
    conn = sqlite3.connect('energy_consumption.db')
    df = pd.read_sql('SELECT * FROM consumption_data', conn)
    conn.close()
    
    if df.empty:
        print("No hay datos para generar gráficas")
        return
    
    # Convertir timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Gráfica de serie temporal con anomalías
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['consumption_kw'], 
             label='Consumo Normal', alpha=0.7, linewidth=1)
    
    anomalies = df[df['final_anomaly'] == 1]
    plt.scatter(anomalies['timestamp'], anomalies['consumption_kw'],
                color='red', label='Anomalías Detectadas', s=50, zorder=5)
    
    plt.title('Consumo Eléctrico y Detección de Anomalías', fontsize=14, fontweight='bold')
    plt.xlabel('Tiempo')
    plt.ylabel('Consumo (kW)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('docs/consumption_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distribución de consumo
    plt.figure(figsize=(10, 6))
    plt.hist(df['consumption_kw'], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(df['consumption_kw'].mean(), color='red', linestyle='--', 
                label=f'Media: {df["consumption_kw"].mean():.2f} kW')
    plt.title('Distribución del Consumo Eléctrico', fontsize=14, fontweight='bold')
    plt.xlabel('Consumo (kW)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/consumption_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Comparación de modelos
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # K-Means scores
    axes[0].hist(df['kmeans_anomaly_score'], bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_title('Puntuaciones de Anomalía - K-Means')
    axes[0].set_xlabel('Distancia al Centroide')
    axes[0].set_ylabel('Frecuencia')
    
    # Isolation Forest resultados
    anomaly_counts = df['isolation_forest_anomaly'].value_counts()
    axes[1].pie(anomaly_counts.values, labels=['Normal', 'Anomalía'], 
                autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    axes[1].set_title('Resultados - Isolation Forest')
    
    plt.tight_layout()
    plt.savefig('docs/models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Patrón diario
    df['hour'] = df['timestamp'].dt.hour
    hourly_pattern = df.groupby('hour')['consumption_kw'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_pattern.index, hourly_pattern.values, 
             marker='o', linewidth=2, markersize=6)
    plt.title('Patrón Diario de Consumo Promedio', fontsize=14, fontweight='bold')
    plt.xlabel('Hora del Día')
    plt.ylabel('Consumo Promedio (kW)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/daily_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráficas generadas exitosamente en la carpeta 'docs'")

def generate_results_table():
    """Genera tabla de resultados"""
    conn = sqlite3.connect('energy_consumption.db')
    df = pd.read_sql('SELECT * FROM consumption_data', conn)
    conn.close()
    
    if df.empty:
        return
    
    # Estadísticas generales
    stats = {
        'Métrica': [
            'Total de Registros',
            'Anomalías Detectadas',
            'Porcentaje de Anomalías',
            'Consumo Promedio (kW)',
            'Consumo Máximo (kW)',
            'Consumo Mínimo (kW)',
            'Desviación Estándar (kW)'
        ],
        'Valor': [
            len(df),
            int(df['final_anomaly'].sum()),
            f"{(df['final_anomaly'].sum() / len(df)) * 100:.2f}%",
            f"{df['consumption_kw'].mean():.2f}",
            f"{df['consumption_kw'].max():.2f}",
            f"{df['consumption_kw'].min():.2f}",
            f"{df['consumption_kw'].std():.2f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('docs/resultados_estadisticos.csv', index=False)
    print("Tabla de resultados generada en 'docs/resultados_estadisticos.csv'")

if __name__ == '__main__':
    generate_all_plots()
    generate_results_table()