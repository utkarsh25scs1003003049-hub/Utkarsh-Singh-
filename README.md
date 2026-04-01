# Utkarsh-Singh-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. DATA ACQUISITION LAYER (Simulated Dataset)
# [span_2](start_span)[span_3](start_span)Based on common fumigants: Methyl Bromide, Chloropicrin, Metam Sodium[span_2](end_span)[span_3](end_span)
data = {
    'fumigant_type': ['Methyl Bromide', 'Chloropicrin', 'Metam Sodium', '1,3-Dichloropropene', 'Methyl Bromide'],
    'soil_type': ['Sand', 'Clay', 'Loam', 'Sand', 'Clay'],
    [span_4](start_span)'moisture_content': [15, 45, 30, 10, 50],  # Measured by IoT sensors[span_4](end_span)
    'distance_to_water_m': [50, 200, 150, 30, 400],
    'is_high_risk': [1, 0, 0, 1, 0] # Target for supervised learning
}

df = pd.DataFrame(data)
# [span_5](start_span)Encode categorical data for processing[span_5](end_span)
df_encoded = pd.get_dummies(df, columns=['fumigant_type', 'soil_type'])

# 2. PROCESSING LAYER: Supervised Learning (Predict Contamination Risk)
X = df_encoded.drop('is_high_risk', axis=1)
y = df_encoded['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [span_6](start_span)Using Random Forest as a representative ML model[span_6](end_span)
risk_model = RandomForestClassifier()
risk_model.fit(X_train, y_train)

# 3. AI AGENT LAYER: Rule-Based Logic System
# [span_7](start_span)Implementing the logic: IF fumigant="X" AND soil="clay" THEN risk="high"[span_7](end_span)
def compliance_agent(fumigant, soil_type):
    """
    [span_8](start_span)An Intelligent Agent that checks regulatory rules autonomously[span_8](end_span).
    """
    if fumigant == "Methyl Bromide" and soil_type == "Sand":
        return "ALERT: High Volatility Risk. Check Buffer Zone Requirements."
    elif soil_type == "Clay":
        return "WARNING: Potential for long-term soil degradation."
    else:
        return "Compliance Check Passed: Standard application procedures apply."

# 4. UNSUPERVISED LEARNING: Identify Hidden Usage Patterns
# [span_9](start_span)Grouping data to find anomalies or clusters in fumigant usage[span_9](end_span)
kmeans = KMeans(n_clusters=2, n_init=10)
df_encoded['usage_cluster'] = kmeans.fit_predict(X)

# 5. [span_10](start_span)RESULTS & FINDINGS[span_10](end_span)
print("--- Soil Fumigant AI/ML Strategy Output ---")
print(f"Risk Prediction Accuracy: {accuracy_score(y_test, risk_model.predict(X_test)) * 100}%")

# Test the Intelligent Agent
test_fumigant = "Methyl Bromide"
test_soil = "Sand"
print(f"\nAgent Analysis for {test_fumigant} in {test_soil} soil:")
print(compliance_agent(test_fumigant, test_soil))

# Summary of Findings
print("\nCluster Analysis (Usage Patterns):")
print(df_encoded[['usage_cluster']].head())
