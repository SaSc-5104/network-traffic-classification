import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


#Split dataset into train/val/test
df = pd.read_csv("data/processed/features.csv")
X = df.drop('label', axis = 1)
y = df['label']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
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
rf = RandomForestClassifier(n_estimators = 200, random_state = 42)
rf.fit(X_train, y_train)

y_pred_val = rf.predict(X_val)
print("Val Accuracy:", accuracy_score(y_val, y_pred_val))
print("Val Macro F1:", f1_score(y_val, y_pred_val, average='macro'))

y_pred_test = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Test Macro F1:", f1_score(y_test, y_pred_test, average='macro'))


#Generate evaluation figures
fig, axes = plt.subplots(1, 2, figsize = (12, 5))
for ax, preds, title in zip(axes, [y_pred_lr, y_pred_test], ["Logistic Regression", "Random Forest"]):
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax = ax, colorbar = False)
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=90)
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi = 150, bbox_inches = 'tight')
plt.close(fig)

importances = pd.Series(rf.feature_importances_, index = X_train.columns)
plt.figure(figsize = (8, 5))
importances.sort_values().plot(kind = 'barh')
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("figures/feature_importances.png", dpi = 150)
plt.close()

label_map = {l:i for i, l in enumerate(sorted(y_test.unique()))}
plt.figure(figsize = (10, 4))
plt.plot([label_map[v] for v in y_test.values], label = "Actual", alpha = 0.7)
plt.plot([label_map[v] for v in y_pred_test], label="Predicted", alpha=0.7)
plt.yticks(range(len(label_map)), label_map.keys())
plt.title("Random Forest: Predicted vs Actual")
plt.legend(); plt.tight_layout()
plt.savefig("figures/rf_predictions.png", dpi=150)
plt.close()


#Print results summary
models = {"Logistic Regression": y_pred_lr, "Random Forest": y_pred_test}
for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average = "macro")
    print(f"{name}: Accuracy={acc:.2f}  Macro F1={f1:.2f}")
    print(classification_report(y_test, preds, zero_division=0))
