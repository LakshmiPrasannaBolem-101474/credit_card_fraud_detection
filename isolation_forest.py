import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
df = pd.read_csv("unlabeled_anomaly_dataset_50000.csv")
print("\nDataset Shape:", df.shape)
print(df.head())
drop_cols = ["transaction_id", "card_number_hash"]

df.drop(
    columns=[c for c in drop_cols if c in df.columns],
    inplace=True,
    errors="ignore"
)

if "transaction_timestamp" in df.columns:

    df["transaction_timestamp"] = pd.to_datetime(
        df["transaction_timestamp"],
        errors="coerce"
    )

    df["hour"] = df["transaction_timestamp"].dt.hour
    df["day"] = df["transaction_timestamp"].dt.day
    df["weekday"] = df["transaction_timestamp"].dt.weekday

    df.drop(columns=["transaction_timestamp"], inplace=True)

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

df_encoded = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=True
)

print("\nEncoded Shape:", df_encoded.shape)

variances = df_encoded.var().sort_values(ascending=False)

important_features = variances.head(15).index
X_main = df_encoded[important_features]

remaining_features = df_encoded.drop(columns=important_features)

pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(remaining_features)

print("\nPCA Components Shape:", X_pca.shape)

X_final = np.hstack([X_main.values, X_pca])

print("Final Feature Vector Shape:", X_final.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

X_train, X_test, idx_train, idx_test = train_test_split(
    X_scaled,
    df_encoded.index,
    test_size=0.30,
    random_state=42
)

print("\nTraining Size:", X_train.shape)
print("Testing Size:", X_test.shape)

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

iso_model.fit(X_train)

test_predictions = iso_model.predict(X_test)

test_fraud_flag = np.where(test_predictions == -1, 1, 0)

test_scores = iso_model.decision_function(X_test)

fraud_percent = (test_fraud_flag.sum() / len(test_fraud_flag)) * 100

print("\nFraud Distribution (Test Data):")
print(pd.Series(test_fraud_flag).value_counts())

print(f"\nDetected Fraud Percentage: {fraud_percent:.2f}%")

plt.figure(figsize=(8,5))

plt.hist(
    test_scores[test_fraud_flag == 0],
    bins=50,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    test_scores[test_fraud_flag == 1],
    bins=50,
    alpha=0.6,
    label="Fraud"
)

plt.legend()
plt.title("Fraud vs Normal Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")

plt.show()

fraud_output = df.loc[idx_test].copy()

fraud_output["fraud_flag"] = test_fraud_flag
fraud_output["anomaly_score"] = test_scores
fraud_output["prediction_label"] = fraud_output["fraud_flag"].map(
    {0: "Normal", 1: "Fraud"}
)

fraud_only = fraud_output[fraud_output["fraud_flag"] == 1]

print("\nNumber of Fraud Transactions Detected:", len(fraud_only))

output_file = "fraud_transactions_only.csv"

fraud_only.to_csv(output_file, index=False)

print(f"\n ONLY Fraud transactions saved to: {output_file}")


print("\n Fraud Detection Completed Successfully.")