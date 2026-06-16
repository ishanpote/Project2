import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_churn_pipeline():
    print("=" * 60)
    print("       TELECOM CUSTOMER CHURN MACHINE LEARNING PIPELINE       ")
    print("=" * 60)
    
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    if not os.path.exists('data'):
        os.makedirs('data')
        
    print("🔄 Loading datasets...")
    df = pd.read_csv(url)
    df.to_csv('data/Telco-Customer-Churn.csv', index=False)
    
    # Preprocessing
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_raw = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    X = pd.get_dummies(X_raw, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🤖 Adjusting Random Forest weights for class imbalance...")
    # Added class_weight='balanced_subsample' to address the low class 1 recall score!
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced_subsample')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- [UPGRADED MODEL EVALUATION LOG] ---")
    print(f"✓ New Test Accuracy: {accuracy * 100:.2f}%")
    print("\n✓ Optimized Classification Metrics:")
    print(classification_report(y_test, y_pred))
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    # Generate Confusion Matrix Asset
    print("📊 Compiling Confusion Matrix Visual asset...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'])
    plt.title('Telecom Churn Prediction Confusion Matrix', fontsize=12, weight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('outputs/churn_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Generate Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(11, 6))
    sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis", hue=X.columns[indices], legend=False)
    plt.title("Top 10 Operational Drivers of Customer Churn Risk", fontsize=14, weight='bold')
    plt.xlabel("Global Model Contribution Importance Score (Gini)")
    plt.ylabel("Telemetry Features")
    plt.tight_layout()
    plt.savefig('outputs/churn_feature_importance.png', dpi=300)
    plt.close()
    
    print("✓ Outputs saved successfully to 'outputs/' directory!")
    print("=" * 60)

if __name__ == "__main__":
    run_churn_pipeline()