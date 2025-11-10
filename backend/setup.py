from setuptools import setup, find_packages

setup(
    name="dace-system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flask>=2.3.3',
        'pandas>=2.0.3',
        'numpy>=1.24.3',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'sqlalchemy>=2.0.20'
    ],
    author="Mauricio Fabian del Real Ramirez, Pedro Misraim Gómez Rodríguez",
    description="Sistema de Detección de Anomalías en Consumo Eléctrico"
)