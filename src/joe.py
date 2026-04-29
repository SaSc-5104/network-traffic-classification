import pandas as pd
import numpy as np
import os
import pyshark

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def extract_features_pcap(file, label):
	print(file)
	cap = pyshark.FileCapture(f'./data/raw/{file}', keep_packets=False)

	times = []
	lengths = []

	for pkt in cap:
		try:
			times.append(float(pkt.sniff_timestamp))
			lengths.append(int(pkt.length))
		except:
			continue

	cap.close()

	intervals = np.diff(times) if len(times) > 1 else []

	return {
		"packet_count": len(times),
		"total_length": np.sum(lengths) if lengths else 0,
		"avg_interval": np.mean(intervals) if len(intervals) > 0 else 0,
		"max_interval": np.max(intervals) if len(intervals) > 0 else 0,
		"min_interval": np.min(intervals) if len(intervals) > 0 else 0,
		"avg_length": np.mean(lengths) if lengths else 0,
		"max_length": np.max(lengths) if lengths else 0,
		"min_length": np.min(lengths) if lengths else 0,
		"most_common_length": pd.Series(lengths).mode()[0] if lengths else 0,
		"label": label
	}


data = []

files = os.listdir('./data/raw')

for file in files:
	if file.endswith(".pcapng"):
		# Extract label from filename (before underscore)
		label = file.split("_")[0].lower()
		data.append(extract_features_pcap(file, label))


df = pd.DataFrame(data)

print("Dataset Preview:")
print(df)
print("\nTotal samples:", len(df))


X = df.drop("label", axis=1)
y = df["label"]


models = {
	"Logistic Regression": LogisticRegression(max_iter=1000),
	"Decision Tree": DecisionTreeClassifier(),
	"Random Forest": RandomForestClassifier()
}


kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
	acc_scores = []
	f1_scores = []

	for train_idx, test_idx in kf.split(X):
		X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
		y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

		model.fit(X_train, y_train)
		preds = model.predict(X_test)

		acc_scores.append(accuracy_score(y_test, preds))
		f1_scores.append(f1_score(y_test, preds, average='macro'))

	results[name] = {
			"accuracy": np.mean(acc_scores),
			"f1": np.mean(f1_scores)
	}


print("\nFINAL RESULTS:\n")

for model_name, scores in results.items():
	print(f"{model_name}:")
	print(f" 	Accuracy: {scores['accuracy']:.4f}")
	print(f" 	F1 Score: {scores['f1']:.4f}")
	print()