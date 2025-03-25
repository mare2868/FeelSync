import unittest
import pandas as pd
import os

# Función de carga de datos (ajustar según tu código)
def load_dataset(file_path):
    """Carga un dataset desde un archivo CSV y lo devuelve como DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    return pd.read_csv(file_path)

class TestDataProcessing(unittest.TestCase):
    def test_load_data(self):
        """Verifica que el dataset de películas se cargue correctamente."""
        file_path = "data/raw/movies.csv"  # Ajusta la ruta según tu estructura
        df = load_dataset(file_path)
        
        self.assertIsInstance(df, pd.DataFrame)  # Verifica que sea un DataFrame
        self.assertGreater(len(df), 0)  # Verifica que tenga datos

if __name__ == '__main__':
    unittest.main()
 
