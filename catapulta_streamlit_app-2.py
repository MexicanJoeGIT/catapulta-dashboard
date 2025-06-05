import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Catapulta.ai Credit Scoring Dashboard")

st.markdown("This dashboard uses synthetic gig economy data to estimate a rider's credit risk and visualize model insights.")

# Train model on synthetic data
np.random.seed(42)
n_samples = 1000

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
st.sidebar.header("Rider Profile Inputs")
weekly_deliveries = st.sidebar.slider("Weekly Deliveries", 0, 100, 30)
avg_rating = st.sidebar.slider("Average Rating (1.0 - 5.0)", 1.0, 5.0, 4.5, 0.1)
days_active_per_week = st.sidebar.slider("Days Active Per Week", 0, 7, 5)
weekly_distance_km = st.sidebar.slider("Weekly Distance (KM)", 0, 500, 150)
smartphone_model_score = st.sidebar.slider("Smartphone Model Score (1=basic, 10=high-end)", 1, 10, 5)
location_stability = st.sidebar.slider("Location Stability (0.0 - 1.0)", 0.0, 1.0, 0.7, 0.01)
fuel_expense_ratio = st.sidebar.slider("Fuel Expense Ratio (0.05 - 0.25)", 0.05, 0.25, 0.15, 0.01)

if st.sidebar.button("Predict Creditworthiness"):
    features = np.array([[weekly_deliveries, avg_rating, days_active_per_week,
                          weekly_distance_km, smartphone_model_score,
                          location_stability, fuel_expense_ratio]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = {
        "Deliveries": weekly_deliveries,
        "Rating": avg_rating,
        "Days Active": days_active_per_week,
        "Distance": weekly_distance_km,
        "Phone Score": smartphone_model_score,
        "Stability": location_stability,
        "Fuel %": fuel_expense_ratio,
        "Approved": bool(prediction),
        "Probability": round(probability, 2)
    }
    st.session_state.history.append(result)

    if prediction == 1:
        st.success(f"✅ Approved: High creditworthiness with a probability of {probability:.2f}")
    else:
        st.error(f"❌ Not approved: Low creditworthiness with a probability of {probability:.2f}")

# Show prediction history
if st.session_state.history:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# Feature importance chart
st.subheader("Model Feature Importance")
importances = model.feature_importances_
features = X.columns
fig, ax = plt.subplots()
ax.barh(features, importances)
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
st.pyplot(fig)

# Optional: Show raw data sample
with st.expander("View Synthetic Training Data"):
    st.dataframe(data.head(20))
