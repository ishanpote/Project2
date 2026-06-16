import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure visualization environment properties
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_churn_pipeline():
    print("=" * 60)
    print("       TELECOM CUSTOMER CHURN MACHINE LEARNING PIPELINE       ")
    print("=" * 60)
    
    # 1. Automated Dataset Ingestion Phase
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    print("🔄 Downloading and staging official IBM Telco customer account ledger...")
    df = pd.read_csv(url)
    df.to_csv('data/Telco-Customer-Churn.csv', index=False)
    
    # 2. Data Cleaning & Feature Engineering Phase
    print("🧹 Cleaning data fields and handling null gaps...")
    # Address empty string anomalies found in the TotalCharges tracking variable
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Target label mapping processing: Yes (Left Company) -> 1, No (Stayed) -> 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Strip identifiers and split feature variables from target classifications
    drop_cols = ['customerID', 'Churn']
    X_raw = df.drop(columns=drop_cols)
    y = df['Churn']
    
    # Perform one-hot encoding across text categories to build mathematical vectors
    X = pd.get_dummies(X_raw, drop_first=True)
    
    # 3. Stratified Data Split Sequence
    # Using stratify ensures our train and validation slices reflect equal target ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Model Training Phase (Random Forest Classifiers)
    print("🤖 Fitting production-grade Random Forest Classifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # 5. Core Model Evaluation Output
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- [MODEL PERFORMANCE MATRIX EVALUATION] ---")
    print(f"✓ Stratified Testing Accuracy Score: {accuracy * 100:.2f}%")
    print("\n✓ Comprehensive Classification Report Summary:")
    print(classification_report(y_test, y_pred))
    
    # 6. Global Feature Importance Analysis (Explainable AI Engine Layer)
    print("\n📊 Isolating critical business drivers via Global Feature Importance...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Grab the top 10 most impactful features
    
    plt.figure(figsize=(11, 6))
    sns.barplot(
        x=importances[indices], 
        y=X.columns[indices], 
        palette="viridis", 
        hue=X.columns[indices], 
        legend=False
    )
    plt.title("Top 10 Operational Drivers of Customer Churn Risk", fontsize=14, weight='bold')
    plt.xlabel("Global Model Contribution Importance Score (Gini)", fontsize=12)
    plt.ylabel("Account Attributes & Telemetry Features", fontsize=12)
    plt.tight_layout()
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    plt.savefig('outputs/churn_feature_importance.png', dpi=300)
    plt.close()
    print("✓ Business impact graph safely stored at: 'outputs/churn_feature_importance.png'")
    print("=" * 60)

if __name__ == "__main__":
    run_churn_pipeline()