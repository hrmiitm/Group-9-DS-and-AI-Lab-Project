from sklearn.ensemble import IsolationForest
import numpy as np


class MetadataAnomalyModel:

    def __init__(self):

        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )

    def train(self, X):

        self.model.fit(X)

    def predict_score(self, X):

        scores = self.model.decision_function(X)

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            normalized = np.zeros_like(scores)
        else:
            normalized = (scores - min_score) / (max_score - min_score)

        anomaly_score = 1 - normalized

        return anomaly_score

# if __name__ == "__main__":

#     import pandas as pd
#     from pathlib import Path

#     print("Running Metadata Anomaly Model Test...")

#     BASE_DIR = Path(__file__).resolve().parents[3]

#     data_path = BASE_DIR / "data" / "raw" / "fake_job_postings.csv"

#     df = pd.read_csv(data_path)

#     df["missing_company_profile"] = df["company_profile"].isna().astype(int)
#     df["missing_salary"] = df["salary_range"].isna().astype(int)
#     df["missing_department"] = df["department"].isna().astype(int)

#     features = [
#         "telecommuting",
#         "has_company_logo",
#         "has_questions",
#         "missing_company_profile",
#         "missing_salary",
#         "missing_department"
#     ]

#     X = df[features]

#     model = MetadataAnomalyModel()

#     model.train(X)

#     scores = model.predict_score(X[:50])

#     print("Sample anomaly scores:")
#     print(scores)