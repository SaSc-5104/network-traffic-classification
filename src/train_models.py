import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#Split dataset into train/val/test
df = pd.read_csv("data/processed/features.csv")
X = df.drop('label', axis = 1)
y = df['label']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_stats = 42, stratify = y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state = 42, stratify = y_temp)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")


#Train Logistic regression model
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter = 1000, random_state = 42) 
lr.fit(X_train_s, y_train)

y_pred_lr = lr.predict(X_test_s)
print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))
print("LR Macro F1:", f1_score(y_test, y_pred_lr, average='macro'))


#Train Random Forest Model (Proposed Model)
rf = RandomForestClassifier(n_estimators = 100, random_stats = 42)
rf.fit(X_train, y_train)

y_pred_val = rf.predict(X_val)
print("Val Accuracy:", accuracy_score(y_val, y_pred_val))
print("Val Macro F1:", f1_score(y_val, y_pred_val, average='macro'))

y_pred_test = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Test Macro F1:", f1_score(y_test, y_pred_test, average='macro'))

