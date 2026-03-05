from pyexpat import features
from sys import flags

from metadata_preprocessing import MetadataPreprocessor
from anomaly_model import MetadataAnomalyModel
from rules_engine import RulesEngine

class MetadataDetector:

    def __init__(self):

        self.preprocessor = MetadataPreprocessor()
        self.model = MetadataAnomalyModel()
        self.rules = RulesEngine()


    def train(self, df):

        features, _ = self.preprocessor.fit_transform(df)

        self.model.train(features)


    def analyze(self, row):

        df_row = row.to_frame().T

        features, rule_features = self.preprocessor.transform(df_row)

        anomaly_score = self.model.predict_score(features)[0]
        rule_result = self.rules.evaluate(rule_features.iloc[0])
        # flags = self.rules.evaluate(rule_features.iloc[0])
        flags = rule_result["flags"]

        # rule_score = min(len(flags) / 5, 1.0)
        rule_score = rule_result["rule_score"]
        final_score = 0.7 * anomaly_score + 0.3 * rule_score

        # risk level
        risk_level = "low"

        if final_score > 0.6:
            risk_level = "high"
        elif final_score > 0.3:
            risk_level = "medium"

        return {
            "metadata_risk_score": float(final_score),
            "anomaly_score": float(anomaly_score),
            "rule_score": float(rule_score),
            "flags": flags,
            "risk_level": risk_level
        }

if __name__ == "__main__":

    import pandas as pd
    from pathlib import Path

    print("Testing MetadataDetector...")

    BASE_DIR = Path(__file__).resolve().parents[3]
    data_path = BASE_DIR / "data" / "raw" / "fake_job_postings.csv"

    df = pd.read_csv(data_path)

    detector = MetadataDetector()

    print("Training metadata anomaly model...")
    detector.train(df)

    print("Running analysis on sample job...")

    result = detector.analyze(df.iloc[0])

    print("\nResult:")
    print(result)