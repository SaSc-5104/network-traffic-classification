import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/features.csv")
X = df.drop('label', axis = 1)
y = df['label']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_stats = 42, stratify = y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state = 42, stratify = y_temp)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
