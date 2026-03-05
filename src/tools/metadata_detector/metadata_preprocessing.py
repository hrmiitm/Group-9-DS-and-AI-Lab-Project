import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class MetadataPreprocessor:

    def __init__(self):
        self.encoder = None
        self.categorical_cols = [
            "employment_type",
            "required_experience",
            "required_education",
            "industry"
        ]

        self.binary_cols = [
            "telecommuting",
            "has_company_logo",
            "has_questions"
        ]

    def clean_data(self, df):

        df = df.copy()
        # create rule-engine features
        df["missing_company_profile"] = df["company_profile"].isna().astype(int)
        df["missing_salary"] = df["salary_range"].isna().astype(int)
        df["missing_department"] = df["department"].isna().astype(int)

        for col in self.categorical_cols:
            df[col] = df[col].fillna("unknown")

        for col in self.binary_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def fit_transform(self, df):

        df = self.clean_data(df)
        
        # columns needed for rule engine
        rule_features = df[[
            "missing_company_profile",
            "missing_salary",
            "missing_department",
            "telecommuting",
            "has_company_logo",
            "has_questions",
            "required_experience"
        ]].copy()

        self.encoder = OneHotEncoder(handle_unknown="ignore")

        encoded = self.encoder.fit_transform(df[self.categorical_cols]).toarray()
        encoded_df = pd.DataFrame(
            encoded,
            index=df.index
        )

        binary = df[self.binary_cols]

        features = pd.concat(
           [binary, encoded_df],            
         axis=1)
        features.columns = features.columns.astype(str)
        return features, rule_features


    def transform(self, df):

        df = self.clean_data(df)
        rule_features = df[[
            "missing_company_profile",
            "missing_salary",
            "missing_department",
            "telecommuting",
            "has_company_logo",
            "has_questions",
            "required_experience"
        ]].copy()

        encoded = self.encoder.transform(df[self.categorical_cols]).toarray()

        encoded_df = pd.DataFrame(
            encoded,
            index=df.index
        )


        binary = df[self.binary_cols]

        features = pd.concat(
           [binary, encoded_df],
         axis=1)
        features.columns = features.columns.astype(str)
        return features, rule_features