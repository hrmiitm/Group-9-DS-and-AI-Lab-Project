import pandas as pd

class RulesEngine:

    def evaluate(self, row):

        score = 0
        flags = []

        if row["missing_company_profile"] == 1:
            score += 0.25
            flags.append("missing_company_profile")

        if row["has_company_logo"] == 0:
            score += 0.20
            flags.append("missing_company_logo")

        if row["telecommuting"] == 1:
            score += 0.15
            flags.append("remote_job")

        if row["missing_salary"] == 1:
            score += 0.15
            flags.append("missing_salary")

        if row["missing_department"] == 1:
            score += 0.10
            flags.append("missing_department")

        if row["required_experience"] == "Entry level":
            score += 0.10
            flags.append("entry_level_job")

        if row["telecommuting"] == 1 and row["missing_company_profile"] == 1:
            score += 0.30
            flags.append("remote_missing_profile")

        if row["telecommuting"] == 1 and row["missing_salary"] == 1:
            score += 0.25
            flags.append("remote_missing_salary")

        return {
            "flags": flags,
            "rule_score": min(score, 1.0)
        }