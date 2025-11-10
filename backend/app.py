from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import sqlite3
from datetime import datetime
from data_simulator import EnergyDataSimulator
from anomaly_detector import AnomalyDetector
import json

app = Flask(__name__)
CORS(app)

# Configuración
DATABASE = 'energy_consumption.db'
detector = AnomalyDetector()
simulator = EnergyDataSimulator()

def index():
    return render_template('index.html') 

def init_database():
    """Inicializa la base de datos"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consumption_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            consumption_kw REAL,
            is_anomaly BOOLEAN,
            kmeans_anomaly_score REAL,
            isolation_forest_anomaly BOOLEAN,
            final_anomaly BOOLEAN,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    """Genera datos simulados"""
    try:
        hours = request.json.get('hours', 168)  # 1 semana por defecto
        
        # Generar datos
        data = simulator.generate_dataset(hours)
        
        # Detectar anomalías
        data_with_anomalies = detector.train_models(data)
        
        # Guardar en base de datos
        conn = sqlite3.connect(DATABASE)
        data_with_anomalies.to_sql('consumption_data', conn, 
                                  if_exists='replace', index=False)
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': f'Datos generados: {len(data)} registros',
            'anomalies_detected': int(data_with_anomalies['final_anomaly'].sum())
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/consumption-data', methods=['GET'])
def get_consumption_data():
    """Obtiene datos de consumo"""
    try:
        conn = sqlite3.connect(DATABASE)
        df = pd.read_sql('SELECT * FROM consumption_data', conn)
        conn.close()
        
        return jsonify({
            'status': 'success',
            'data': df.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtiene estadísticas del sistema"""
    try:
        conn = sqlite3.connect(DATABASE)
        df = pd.read_sql('SELECT * FROM consumption_data', conn)
        conn.close()
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No hay datos'})
            
        stats = {
            'total_records': len(df),
            'total_anomalies': int(df['final_anomaly'].sum()),
            'anomaly_percentage': round((df['final_anomaly'].sum() / len(df)) * 100, 2),
            'avg_consumption': round(df['consumption_kw'].mean(), 2),
            'max_consumption': round(df['consumption_kw'].max(), 2),
            'min_consumption': round(df['consumption_kw'].min(), 2)
        }
        
        return jsonify({'status': 'success', 'stats': stats})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/evaluate', methods=['GET'])
def evaluate_models():
    """Evalúa los modelos"""
    try:
        conn = sqlite3.connect(DATABASE)
        df = pd.read_sql('SELECT * FROM consumption_data', conn)
        conn.close()
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No hay datos'})
            
        evaluation = detector.evaluate_models(df)
        
        return jsonify({
            'status': 'success', 
            'evaluation': evaluation
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    init_database()
    app.run(debug=True, host='0.0.0.0', port=5000)