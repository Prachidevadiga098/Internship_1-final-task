import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import numpy as np

# Load CSV files
orders_path = r'C:\Users\Prachitas\OneDrive\Documents\project1\orders.csv'
returns_path = r'C:\Users\Prachitas\OneDrive\Documents\project1\returns.csv'

orders = pd.read_csv(orders_path)
returns = pd.read_csv(returns_path)

# Clean Data
orders.dropna(inplace=True)
returns.dropna(inplace=True)
orders.drop_duplicates(inplace=True)
returns.drop_duplicates(inplace=True)

orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
returns['return_date'] = pd.to_datetime(returns['return_date'], errors='coerce')
orders.dropna(subset=['order_date'], inplace=True)
returns.dropna(subset=['return_date'], inplace=True)

orders['product_id'] = orders['product_id'].astype(str)
returns['product_id'] = returns['product_id'].astype(str)

# Merge and Add Flag
merged = pd.merge(orders, returns, on='order_id', how='left')
merged['is_returned'] = merged['return_date'].notnull().astype(int)

# Save Merged File
merged.to_csv(r'C:\Users\Prachitas\OneDrive\Documents\project1\merged_orders_returns.csv', index=False)
print("\n✅ Merged data saved!")

# --- Return Rate Function ---
def calculate_return_rate(df, group_col):
    if group_col not in df.columns:
        print(f"⚠️ Column '{group_col}' not found.")
        return None
    result = df.groupby(group_col)['is_returned'].mean().reset_index()
    result.columns = [group_col, 'return_rate (%)']
    result['return_rate (%)'] *= 100
    return result

# --- Analyze Return Rates ---
for col in ['category', 'supplier', 'region', 'marketing_channel']:
    if col in merged.columns:
        print(f"\n📊 Return Rate by {col.capitalize()}:")
        print(calculate_return_rate(merged, col))

# --- Logistic Regression Model ---
# Ensure price is numeric
merged['price'] = pd.to_numeric(merged['price'], errors='coerce')
merged.dropna(subset=['price'], inplace=True)

# Select Features
feature_cols = ['price', 'category', 'supplier']
if not all(col in merged.columns for col in feature_cols):
    raise ValueError("❌ One or more required columns missing: price, category, supplier")

X = merged[feature_cols]
y = merged['is_returned']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Check class balance
if len(y.unique()) < 2:
    raise ValueError("❌ Not enough class variety in 'is_returned'. Need both 0 and 1.")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Fit Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict Probabilities
merged['return_probability'] = model.predict_proba(X_encoded)[:, 1]

# Save High-Risk Products
high_risk = merged[merged['return_probability'] > 0.5]
high_risk.to_csv(r'C:\Users\Prachitas\OneDrive\Documents\project1\high_risk_products.csv', index=False)
print(f"\n✅ High-risk products saved! ({len(high_risk)} rows)")

# Show Top Predictions
print("\n🔝 Top 5 by Return Probability:")
print(merged[['order_id', 'return_probability']].sort_values(by='return_probability', ascending=False).head())
merged.to_csv(r'C:\Users\Prachitas\OneDrive\Documents\project1\merged_orders_with_risk.csv', index=False)