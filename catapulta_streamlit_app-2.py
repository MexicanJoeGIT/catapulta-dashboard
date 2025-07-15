# Install dependencies (if not already in Colab)
!pip install -q scikit-learn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic data
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

# Train model
X = data.drop('repaid_on_time', axis=1)
y = data['repaid_on_time']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- USER INPUTS ---
print("Enter gig worker profile:")
username = input("Rider ID or Username: ")
weekly_deliveries = int(input("Weekly Deliveries: "))
avg_rating = float(input("Average Rating (1.0 - 5.0): "))
days_active_per_week = int(input("Days Active Per Week (0–7): "))
weekly_distance_km = float(input("Weekly Distance (KM): "))
smartphone_model_score = int(input("Smartphone Model Score (1–10): "))
location_stability = float(input("Location Stability (0.0–1.0): "))
fuel_expense_ratio = float(input("Fuel Expense Ratio (0.05–0.25): "))

# --- PREDICTION ---
features = np.array([[weekly_deliveries, avg_rating, days_active_per_week,
                      weekly_distance_km, smartphone_model_score,
                      location_stability, fuel_expense_ratio]])
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

print(f"\nRider: {username}")
if prediction == 1:
    print(f"✅ Approved for {username}: High creditworthiness with a probability of {probability:.2f}")
else:
    print(f"❌ Not approved for {username}: Low creditworthiness with a probability of {probability:.2f}")

