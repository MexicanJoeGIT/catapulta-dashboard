import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide")
# Apply white background
st.markdown("""
    <style>
    .main {
        background-color: light green;
    }
    </style>
""", unsafe_allow_html=True)
# Display logo
logo = Image.open("logo.png")
st.image(logo, width=360)

st.title("Catapulta.ai Credit Scoring Dashboard")

st.markdown("Este dashboard utiliza datos de la industria de delivery para estimar el riesgo crediticio de un rider y visualizar información del modelo.")

# Train model on synthetic data
np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    'weekly_deliveries': np.random.poisson(lam=30, size=n_samples),
    'avg_rating': np.clip(np.random.normal(loc=4.5, scale=0.3, size=n_samples), 1.0, 5.0),
    'days_active_per_week': np.random.randint(3, 7, size=n_samples),
    'weekly_distance_km': np.random.normal(loc=150, scale=30, size=n_samples),
    'smartphone_model_score': np.random.randint(1, 10, size=n_samples),
    'location_stability': np.random.beta(2, 1, size=n_samples),
    'fuel_expense_ratio': np.random.uniform(0.05, 0.25, size=n_samples),
})

data['repaid_on_time'] = (
    (data['weekly_deliveries'] > 25) &
    (data['avg_rating'] > 4.2) &
    (data['days_active_per_week'] >= 4) &
    (data['location_stability'] > 0.5)
).astype(int)

X = data.drop('repaid_on_time', axis=1)
y = data['repaid_on_time']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialize session state to track prediction history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar inputs
st.sidebar.header("Inputs Perfil del Rider")
print("Enter gig worker profile:")
username = input("Rider ID o Username: ")
weekly_deliveries = st.sidebar.slider("Entregas Semanales", 0, 100, 30)
avg_rating = st.sidebar.slider("Calificación Promedio (1.0 - 5.0)", 1.0, 5.0, 4.5, 0.1)
days_active_per_week = st.sidebar.slider("Dias Activos por Semana", 0, 7, 5)
weekly_distance_km = st.sidebar.slider("Distancia recorrida semanalmente (KM)", 0, 500, 150)
smartphone_model_score = st.sidebar.slider("Puntaje Smartphone (1=basico, 10=high-end)", 1, 10, 5)
location_stability = st.sidebar.slider("Ubicación estable (0.0 - 1.0)", 0.0, 1.0, 0.7, 0.01)
fuel_expense_ratio = st.sidebar.slider("Gasto en gasolina vs Ingresos (0.05 - 0.25)", 0.05, 0.25, 0.15, 0.01)

if st.sidebar.button("Predecir solvencia crediticia"):
    features = np.array([[weekly_deliveries, avg_rating, days_active_per_week,
                          weekly_distance_km, smartphone_model_score,
                          location_stability, fuel_expense_ratio]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = {
        "Entregas": weekly_deliveries,
        "Rating": avg_rating,
        "Dias": days_active_per_week,
        "Distancia": weekly_distance_km,
        "Smartphone": smartphone_model_score,
        "Estabilidad Loc": location_stability,
        "Gas %": fuel_expense_ratio,
        "Aprobado": bool(prediction),
        "Probabilidad": round(probability, 2)
    }
    st.session_state.history.append(result)

    if prediction == 1:
        st.success(f"✅ Aprobado: Alta solvencia crediticia con probabilidad de pago de{probability:.2f}")
    else:
        st.error(f"❌ NO Aprobado: Baja solvencia crediticia con probabilidad de pago de {probability:.2f}")

# Show prediction history
if st.session_state.history:
    st.subheader("Historial de predicciones")
    st.dataframe(pd.DataFrame(st.session_state.history))

# Feature importance chart (simplified with short names)
st.subheader("Top Predictors")
importances = model.feature_importances_
features = [
    "Entregas", "Rating", "Dias", "Distancia", "Smartphone", "Estabilidad Loc", "Gas %"
]
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))

# Explanation section
with st.expander("📘 Explicación Inputs Rider"):
    st.markdown("""
| Input | Que mide | Porque importa |
|---------|------------------|-----------------|
| Entregas Semanales | Numero de entregas completadas en una semana | Indica productividad y capacidad de generación de ingreso |
| Rating Promedio| Rating promedio del cliente | Refleja confianza y calidad de seervicio |
| Dias activo por semana| Frecuancia de trabajo semanal | Un mayor compromiso sugiere ingresos estables|
| Distancia recorrida semanalmente | Kilometros recorridos semanalmente| Proxy de volumen de trabajo y compromiso |
| Puntaje Smartphone|Calidad de Hardware (1=basico, 10=high-end) | Indica preparación digital y profesionalismo |
| Estabilidad Ubicación| Consistencia de la zona de trabajo | Las zonas estables significan clientes recurrentes y eficiencia de ruta |
| Gasto en gasolina | % de ingresos gastados en combustible| Los ratios altos pueden indicar márgenes bajos y estrés financiero |
    """)

# Optional: Show raw data sample
with st.expander("📊 View Synthetic Training Data"):
    st.dataframe(data.head(20))


