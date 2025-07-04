import pandas as pd
import os

# Calcular ruta raíz del proyecto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "housing.csv")
URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

def download_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df = pd.read_csv(URL)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Dataset guardado en: {DATA_PATH} ({df.shape[0]} filas)")

if __name__ == "__main__":
    download_data()

