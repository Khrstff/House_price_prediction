# 🏡 House Price Predictor

Este proyecto es una aplicación de Machine Learning interactiva desarrollada con **Python**, **scikit-learn** y **Streamlit**, que permite predecir el precio estimado de una vivienda en función de características como ubicación, número de habitaciones, antigüedad y más.

> ⚠️ *Este modelo fue entrenado con datos reales del estado de California (EE.UU.). Las predicciones para otros países como México deben tomarse únicamente con fines demostrativos.*

---

## 📂 Estructura del Proyecto

```
house-price-predictor/
├── app/
│   └── streamlit_app.py        # App interactiva
├── data/
│   └── housing.csv             # Dataset original
├── models/
│   ├── linear_model.pkl        # Modelo entrenado
│   └── scaler.pkl              # Escalador de features
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
│   ├── download_data.py        # Descarga el dataset
│   └── train_model.py          # Entrena el modelo
├── requirements.txt
└── README.md
```

---

## 📈 Dataset

Utiliza el conjunto de datos de precios de viviendas de California proporcionado por [Aurélien Géron](https://github.com/ageron/handson-ml) en su libro *Hands-On Machine Learning*.

- 📅 Fuente: [https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv)
- Variables:
  - `median_income`, `total_rooms`, `population`, `ocean_proximity`, etc.
  - Objetivo: `median_house_value` (precio en USD)

---

## 🛠️ Tecnologías utilizadas

- Python 3.10+
- Pandas, NumPy
- scikit-learn (Regresión lineal)
- Streamlit
- Matplotlib, Seaborn (EDA)
- Joblib

---

## ⚙️ Instalación y ejecución

1. Clona el repositorio:

```bash
git clone https://github.com/tuusuario/house-price-predictor.git
cd house-price-predictor
```

2. Crea un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instala dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Descarga el dataset:

```bash
python3 src/download_data.py
```

5. Entrena el modelo:

```bash
python3 src/train_model.py
```

6. Ejecuta la app:

```bash
streamlit run app/streamlit_app.py
```

---

## 📌 Ejemplo de predicción

Puedes ingresar manualmente:

- Coordenadas (latitud, longitud)
- Habitaciones totales
- Ingreso medio
- Proximidad al océano
- etc.

Y la app devolverá un precio estimado en **USD**.

---

## ⚠️ Limitaciones

- El modelo fue entrenado únicamente con datos del estado de California.
- Las predicciones para viviendas en México u otros países son aproximaciones sin validez comercial.

---


## 🤝 Autor

Desarrollado por [KHRSTFF](https://github.com/Khrstff)\
Contáctame si deseas adaptar este modelo a datos locales o explorar más casos reales.

