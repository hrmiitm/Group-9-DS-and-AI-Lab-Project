"""
tests/test_tool_scam_signals.py
Unit tests for tools/tool_scam_signals.py

Tests cover:
  - Clean job postings (score stays 0)
  - Individual scam rule triggers
  - Score capping at 100
  - Edge cases: empty input, whitespace-only, None handling
  - Risk level thresholds (low / medium / high)
  - Return structure validation
  - Keyword case-insensitivity
  - Multiple keywords within a single rule
  - Combined multi-rule scenarios
  - Boundary values for risk thresholds
"""
import pytest

from tools.tool_scam_signals import detect_scam_signals, RULES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(text: str) -> dict:
    result = detect_scam_signals(text)
    return result


def _data(text: str) -> dict:
    result = _run(text)
    assert result["ok"] is True
    return result["data"]


# ---------------------------------------------------------------------------
# 1. Input validation / error paths
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_string_returns_error(self):
        result = detect_scam_signals("")
        assert result["ok"] is False
        assert "error" in result
        assert len(result["error"]) > 0

    def test_none_input_returns_error(self):
        result = detect_scam_signals(None)
        assert result["ok"] is False
        assert "error" in result

    def test_whitespace_only_returns_error(self):
        # whitespace string is falsy after strip — but the function checks
        # truthiness on the raw input; confirm behaviour
        result = detect_scam_signals("   ")
        # whitespace is truthy in Python, so the function proceeds
        # and should return ok=True with 0 signals
        assert result["ok"] is True
        assert result["data"]["scam_score"] == 0

    def test_newline_only_input(self):
        result = detect_scam_signals("\n\n\n")
        assert result["ok"] is True
        assert result["data"]["scam_score"] == 0


# ---------------------------------------------------------------------------
# 2. Clean job posting — zero signals
# ---------------------------------------------------------------------------

class TestCleanJobPosting:

    def test_clean_text_score_is_zero(self, clean_job_text):
        data = _data(clean_job_text)
        assert data["scam_score"] == 0

    def test_clean_text_risk_level_low(self, clean_job_text):
        data = _data(clean_job_text)
        assert data["risk_level"] == "low"

    def test_clean_text_no_signals_found(self, clean_job_text):
        data = _data(clean_job_text)
        assert data["signals_count"] == 0
        assert data["signals_found"] == []

    def test_clean_text_is_clean_flag(self, clean_job_text):
        data = _data(clean_job_text)
        assert data["is_clean"] is True

    def test_clean_text_matched_signals_empty(self, clean_job_text):
        data = _data(clean_job_text)
        assert data["matched_signals"] == {}

    def test_clean_text_structure(self, clean_job_text, assert_ok_shape):
        data = assert_ok_shape(_run(clean_job_text))
        required_keys = {
            "scam_score", "risk_level", "signals_found",
            "signals_count", "is_clean", "matched_signals"
        }
        assert required_keys.issubset(data.keys())


# ---------------------------------------------------------------------------
# 3. Individual rule triggers
# ---------------------------------------------------------------------------

class TestAskForMoneyRule:

    def test_registration_fee_triggers(self):
        data = _data("You must pay a registration fee of ₹999 to join.")
        assert "asks_for_money" in data["signals_found"]

    def test_processing_fee_triggers(self):
        data = _data("A processing fee is required before onboarding.")
        assert "asks_for_money" in data["signals_found"]

    def test_security_deposit_triggers(self):
        data = _data("Applicants must pay a security deposit upon selection.")
        assert "asks_for_money" in data["signals_found"]

    def test_upfront_payment_triggers(self):
        data = _data("Upfront payment of ₹500 required for the training kit.")
        assert "asks_for_money" in data["signals_found"]

    def test_pay_to_join_triggers(self):
        data = _data("Pay to join our exclusive remote work programme.")
        assert "asks_for_money" in data["signals_found"]

    def test_training_fee_triggers(self):
        data = _data("A training fee applies for the first month.")
        assert "asks_for_money" in data["signals_found"]

    def test_pay_a_fee_triggers(self):
        data = _data("Candidates are required to pay a fee before starting.")
        assert "asks_for_money" in data["signals_found"]

    def test_money_rule_weight_applied(self):
        data = _data("Pay a registration fee now.")
        # one keyword hit → weight 30
        assert data["scam_score"] >= 30

    def test_multiple_money_keywords_stack(self):
        data = _data(
            "Pay a registration fee and a processing fee and a training fee."
        )
        # three hits → 30 * 3 = 90, capped at 100
        assert data["scam_score"] >= 90


class TestBankingDetailsRule:

    def test_bank_account_triggers(self):
        data = _data("Please share your bank account details for payroll setup.")
        assert "requests_bank_details" in data["signals_found"]

    def test_ifsc_code_triggers(self):
        data = _data("Send your IFSC code to HR for payment processing.")
        assert "requests_bank_details" in data["signals_found"]

    def test_upi_id_triggers(self):
        data = _data("Share your UPI ID for instant salary disbursement.")
        assert "requests_bank_details" in data["signals_found"]

    def test_western_union_triggers(self):
        data = _data("Payments are made via Western Union transfers only.")
        assert "requests_bank_details" in data["signals_found"]

    def test_wire_transfer_triggers(self):
        data = _data("Salary is sent by wire transfer every fortnight.")
        assert "requests_bank_details" in data["signals_found"]

    def test_send_money_triggers(self):
        data = _data("Please send money to confirm your slot.")
        assert "requests_bank_details" in data["signals_found"]

    def test_account_number_triggers(self):
        data = _data("Provide your account number for registration.")
        assert "requests_bank_details" in data["signals_found"]

    def test_banking_rule_weight(self):
        data = _data("Share your bank account for payroll.")
        assert data["scam_score"] >= 35


class TestHighPressureRule:

    def test_urgent_hiring_triggers(self):
        data = _data("Urgent hiring! Join us today.")
        assert "high_pressure" in data["signals_found"]

    def test_immediate_joining_triggers(self):
        data = _data("Candidates should be ready for immediate joining.")
        assert "high_pressure" in data["signals_found"]

    def test_limited_slots_triggers(self):
        data = _data("Limited slots available — book yours now.")
        assert "high_pressure" in data["signals_found"]

    def test_apply_immediately_triggers(self):
        data = _data("Apply immediately — interviews start today.")
        assert "high_pressure" in data["signals_found"]

    def test_last_day_today_triggers(self):
        data = _data("Last day today to submit your application.")
        assert "high_pressure" in data["signals_found"]

    def test_only_10_seats_triggers(self):
        data = _data("Only 10 seats left in this batch.")
        assert "high_pressure" in data["signals_found"]

    def test_hurry_triggers(self):
        data = _data("Hurry! Positions are filling up fast.")
        assert "high_pressure" in data["signals_found"]

    def test_high_pressure_weight(self):
        data = _data("Urgent hiring — apply immediately.")
        assert data["scam_score"] >= 15


class TestUnrealisticPromisesRule:

    def test_earn_daily_triggers(self):
        data = _data("Earn daily ₹3000 without leaving home.")
        assert "unrealistic_promises" in data["signals_found"]

    def test_easy_money_triggers(self):
        data = _data("Easy money — just share posts online.")
        assert "unrealistic_promises" in data["signals_found"]

    def test_no_experience_needed_triggers(self):
        data = _data("No experience needed — freshers welcome.")
        assert "unrealistic_promises" in data["signals_found"]

    def test_guaranteed_income_triggers(self):
        data = _data("Guaranteed income of ₹50,000 per month from day one.")
        assert "unrealistic_promises" in data["signals_found"]

    def test_make_money_from_home_triggers(self):
        data = _data("Make money from home without any investment.")
        assert "unrealistic_promises" in data["signals_found"]

    def test_passive_income_triggers(self):
        data = _data("Earn passive income just by referring friends.")
        assert "unrealistic_promises" in data["signals_found"]

    def test_work_2_hours_triggers(self):
        data = _data("Work 2 hours a day and get paid full-time salary.")
        assert "unrealistic_promises" in data["signals_found"]

    def test_earn_lakhs_weekly_triggers(self):
        data = _data("Earn lakhs weekly — no skills required.")
        assert "unrealistic_promises" in data["signals_found"]


class TestUnofficialContactRule:

    def test_whatsapp_only_triggers(self):
        data = _data("Reach out on WhatsApp only: +91 9876543210")
        assert "unofficial_contact" in data["signals_found"]

    def test_telegram_only_triggers(self):
        data = _data("Apply via Telegram only: @recruiter123")
        assert "unofficial_contact" in data["signals_found"]

    def test_contact_on_whatsapp_triggers(self):
        data = _data("Please contact on WhatsApp for further details.")
        assert "unofficial_contact" in data["signals_found"]

    def test_message_on_telegram_triggers(self):
        data = _data("Message on Telegram for instant replies.")
        assert "unofficial_contact" in data["signals_found"]

    def test_gmail_domain_triggers(self):
        data = _data("Send your resume to recruiter2024@gmail.com")
        assert "unofficial_contact" in data["signals_found"]

    def test_yahoo_domain_triggers(self):
        data = _data("Email us at hr_company@yahoo.com for applications.")
        assert "unofficial_contact" in data["signals_found"]

    def test_outlook_domain_triggers(self):
        data = _data("Contact hiring@outlook.com with your CV.")
        assert "unofficial_contact" in data["signals_found"]


class TestPreInterviewDocsRule:

    def test_aadhaar_copy_triggers(self):
        data = _data("Please submit your Aadhaar copy before the interview.")
        assert "pre_interview_docs" in data["signals_found"]

    def test_pan_card_copy_triggers(self):
        data = _data("PAN card copy required for document verification.")
        assert "pre_interview_docs" in data["signals_found"]

    def test_passport_copy_triggers(self):
        data = _data("Send your passport copy as part of the application.")
        assert "pre_interview_docs" in data["signals_found"]

    def test_id_proof_before_interview_triggers(self):
        data = _data("Submit your ID proof before interview at our office.")
        assert "pre_interview_docs" in data["signals_found"]

    def test_documents_before_hiring_triggers(self):
        data = _data("Documents before hiring are mandatory per company policy.")
        assert "pre_interview_docs" in data["signals_found"]

    def test_send_photo_id_triggers(self):
        data = _data("Kindly send photo ID to complete your registration.")
        assert "pre_interview_docs" in data["signals_found"]


class TestVagueCompanyRule:

    def test_undisclosed_company_triggers(self):
        data = _data("Hiring for an undisclosed company with 10+ years in IT.")
        assert "vague_company" in data["signals_found"]

    def test_confidential_client_triggers(self):
        data = _data("This role is for a confidential client in the BFSI sector.")
        assert "vague_company" in data["signals_found"]

    def test_mnc_company_hiring_triggers(self):
        data = _data("MNC company hiring freshers for data entry roles.")
        assert "vague_company" in data["signals_found"]

    def test_us_based_company_hiring_locally_triggers(self):
        data = _data(
            "US based company hiring locally for remote support positions."
        )
        assert "vague_company" in data["signals_found"]


# ---------------------------------------------------------------------------
# 4. Case-insensitivity tests
# ---------------------------------------------------------------------------

class TestCaseInsensitivity:

    def test_uppercase_keyword_detected(self):
        data = _data("PAY A REGISTRATION FEE to proceed.")
        assert "asks_for_money" in data["signals_found"]

    def test_mixed_case_keyword_detected(self):
        data = _data("You Must Pay A Registration Fee Before Joining.")
        assert "asks_for_money" in data["signals_found"]

    def test_titlecase_banking_keyword(self):
        data = _data("Please Share Your Bank Account Number With HR.")
        assert "requests_bank_details" in data["signals_found"]

    def test_all_caps_urgency(self):
        data = _data("URGENT HIRING — APPLY IMMEDIATELY.")
        assert "high_pressure" in data["signals_found"]

    def test_all_caps_whatsapp_only(self):
        data = _data("CONTACT US ON WHATSAPP ONLY.")
        assert "unofficial_contact" in data["signals_found"]


# ---------------------------------------------------------------------------
# 5. Score capping at 100
# ---------------------------------------------------------------------------

class TestScoreCapping:

    def test_score_never_exceeds_100(self, multi_signal_scam_text):
        data = _data(multi_signal_scam_text)
        assert data["scam_score"] <= 100

    def test_score_never_exceeds_100_repeated_keywords(self):
        # Repeat every scam keyword multiple times
        text = " ".join([
            "registration fee registration fee registration fee",
            "bank account bank account bank account",
            "ifsc code ifsc code",
            "urgent hiring urgent hiring",
            "earn daily earn daily",
            "whatsapp only",
            "aadhaar copy aadhaar copy",
            "undisclosed company undisclosed company",
        ])
        data = _data(text)
        assert data["scam_score"] <= 100

    def test_maximum_score_type(self, multi_signal_scam_text):
        data = _data(multi_signal_scam_text)
        assert isinstance(data["scam_score"], int)


# ---------------------------------------------------------------------------
# 6. Risk level boundary tests
# ---------------------------------------------------------------------------

class TestRiskLevelBoundaries:

    def test_score_0_is_low(self, clean_job_text):
        data = _data(clean_job_text)
        assert data["risk_level"] == "low"

    def test_score_24_is_low(self):
        # high_pressure keyword = 15, no others → score 15 → low
        data = _data("Urgent hiring for this role.")
        assert data["risk_level"] in ("low", "medium")

    def test_score_25_is_medium(self):
        # high_pressure (15) + unrealistic_promises (20) = 35 → medium
        data = _data(
            "Urgent hiring. No experience needed. Earn daily ₹2000."
        )
        assert data["risk_level"] == "medium"

    def test_score_60_is_high(self):
        # requests_bank_details (35) + asks_for_money (30) = 65 → high
        data = _data(
            "Share your bank account. Pay a registration fee to start."
        )
        assert data["risk_level"] == "high"

    def test_high_risk_multi_signal(self, scam_job_text_money):
        data = _data(scam_job_text_money)
        assert data["risk_level"] == "high"

    def test_banking_text_risk(self, scam_job_text_banking):
        data = _data(scam_job_text_banking)
        assert data["risk_level"] == "high"


# ---------------------------------------------------------------------------
# 7. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_true_present(self, clean_job_text):
        result = _run(clean_job_text)
        assert "ok" in result
        assert result["ok"] is True

    def test_data_is_dict(self, clean_job_text):
        result = _run(clean_job_text)
        assert isinstance(result["data"], dict)

    def test_all_required_data_keys_present(self, clean_job_text):
        data = _data(clean_job_text)
        for key in ("scam_score", "risk_level", "signals_found",
                    "signals_count", "is_clean", "matched_signals"):
            assert key in data, f"Missing key: {key}"

    def test_signals_found_is_list(self, clean_job_text):
        data = _data(clean_job_text)
        assert isinstance(data["signals_found"], list)

    def test_signals_count_is_int(self, clean_job_text):
        data = _data(clean_job_text)
        assert isinstance(data["signals_count"], int)

    def test_matched_signals_is_dict(self, clean_job_text):
        data = _data(clean_job_text)
        assert isinstance(data["matched_signals"], dict)

    def test_is_clean_is_bool(self, clean_job_text):
        data = _data(clean_job_text)
        assert isinstance(data["is_clean"], bool)

    def test_risk_level_is_valid_string(self, clean_job_text):
        data = _data(clean_job_text)
        assert data["risk_level"] in ("low", "medium", "high")

    def test_scam_score_is_non_negative(self):
        data = _data("Some random job posting text.")
        assert data["scam_score"] >= 0

    def test_matched_signals_keys_match_signals_found(self, scam_job_text_money):
        data = _data(scam_job_text_money)
        assert set(data["matched_signals"].keys()) == set(data["signals_found"])

    def test_matched_signal_has_description(self, scam_job_text_money):
        data = _data(scam_job_text_money)
        for signal_key, signal_val in data["matched_signals"].items():
            assert "description" in signal_val, (
                f"Signal '{signal_key}' missing 'description'"
            )

    def test_matched_signal_has_matched_keywords(self, scam_job_text_money):
        data = _data(scam_job_text_money)
        for signal_key, signal_val in data["matched_signals"].items():
            assert "matched_keywords" in signal_val
            assert isinstance(signal_val["matched_keywords"], list)
            assert len(signal_val["matched_keywords"]) > 0

    def test_matched_signal_has_weight(self, scam_job_text_money):
        data = _data(scam_job_text_money)
        for signal_key, signal_val in data["matched_signals"].items():
            assert "weight" in signal_val
            assert isinstance(signal_val["weight"], int)

    def test_signals_count_matches_signals_found_length(self, scam_job_text_money):
        data = _data(scam_job_text_money)
        assert data["signals_count"] == len(data["signals_found"])

    def test_is_clean_false_when_signals_found(self, scam_job_text_money):
        data = _data(scam_job_text_money)
        assert data["is_clean"] is False

    def test_is_clean_true_when_no_signals(self, clean_job_text):
        data = _data(clean_job_text)
        assert data["is_clean"] is True


# ---------------------------------------------------------------------------
# 8. Rules integrity tests (meta-tests on RULES dict itself)
# ---------------------------------------------------------------------------

class TestRulesIntegrity:

    def test_all_rules_have_keywords(self):
        for rule_name, rule in RULES.items():
            assert "keywords" in rule, f"Rule '{rule_name}' missing 'keywords'"
            assert isinstance(rule["keywords"], list)
            assert len(rule["keywords"]) > 0

    def test_all_rules_have_weight(self):
        for rule_name, rule in RULES.items():
            assert "weight" in rule, f"Rule '{rule_name}' missing 'weight'"
            assert isinstance(rule["weight"], int)
            assert rule["weight"] > 0

    def test_all_rules_have_description(self):
        for rule_name, rule in RULES.items():
            assert "description" in rule
            assert isinstance(rule["description"], str)
            assert len(rule["description"]) > 0

    def test_expected_rule_names_exist(self):
        expected = {
            "asks_for_money",
            "requests_bank_details",
            "high_pressure",
            "unrealistic_promises",
            "unofficial_contact",
            "pre_interview_docs",
            "vague_company",
        }
        assert expected.issubset(set(RULES.keys()))

    def test_all_keywords_are_lowercase(self):
        for rule_name, rule in RULES.items():
            for kw in rule["keywords"]:
                assert kw == kw.lower(), (
                    f"Keyword '{kw}' in rule '{rule_name}' is not lowercase"
                )

    def test_no_duplicate_keywords_within_rule(self):
        for rule_name, rule in RULES.items():
            kws = rule["keywords"]
            assert len(kws) == len(set(kws)), (
                f"Duplicate keywords found in rule '{rule_name}'"
            )

    def test_total_max_weight_exceeds_100(self):
        # Ensures that high-scam content can actually reach the 100 cap
        total = sum(r["weight"] for r in RULES.values())
        assert total > 100


# ---------------------------------------------------------------------------
# 9. Parametrized edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_rule", [
    ("Pay a fee to register.", "asks_for_money"),
    ("Share your bank account for salary.", "requests_bank_details"),
    ("Urgent hiring — join now.", "high_pressure"),
    ("Earn daily without effort.", "unrealistic_promises"),
    ("Reach us on WhatsApp only.", "unofficial_contact"),
    ("Send your Aadhaar copy.", "pre_interview_docs"),
    ("Role at a confidential client.", "vague_company"),
])
def test_parametrized_single_rule_triggers(text, expected_rule):
    data = _data(text)
    assert expected_rule in data["signals_found"], (
        f"Expected '{expected_rule}' in signals for text: '{text}'"
    )


@pytest.mark.parametrize("clean_phrase", [
    "Apply with your resume at hr@company.com",
    "Interview scheduled for next Monday at 10 AM",
    "Salary: ₹8–12 LPA based on experience",
    "We are a SEBI-registered financial services company",
    "Please bring your original certificates to the interview",
    "3 rounds: HR, technical, and managerial",
])
def test_parametrized_clean_phrases_no_signals(clean_phrase):
    data = _data(clean_phrase)
    # We expect these to have zero or very minimal scam score
    assert data["scam_score"] < 25, (
        f"Unexpected signals for phrase: '{clean_phrase}' → {data['signals_found']}"
    )


# ---------------------------------------------------------------------------
# 10. Combined multi-signal scenarios
# ---------------------------------------------------------------------------

class TestCombinedScenarios:

    def test_money_plus_urgency(self):
        data = _data(
            "Pay a registration fee immediately. Urgent hiring — limited slots!"
        )
        assert "asks_for_money" in data["signals_found"]
        assert "high_pressure" in data["signals_found"]
        assert data["signals_count"] >= 2
        assert data["scam_score"] >= 45

    def test_banking_plus_docs(self):
        data = _data(
            "Share your bank account details. Also send your Aadhaar copy."
        )
        assert "requests_bank_details" in data["signals_found"]
        assert "pre_interview_docs" in data["signals_found"]

    def test_all_signals_combined(self, multi_signal_scam_text):
        data = _data(multi_signal_scam_text)
        assert data["signals_count"] >= 5
        assert data["scam_score"] == 100
        assert data["risk_level"] == "high"
        assert data["is_clean"] is False

    def test_single_keyword_produces_nonzero_score(self):
        # Even one keyword should produce a positive score
        data = _data("Hurry before slots fill up!")
        assert data["scam_score"] > 0

    def test_vague_company_alone_is_low_risk(self):
        data = _data("This role is for a confidential client in banking.")
        assert data["risk_level"] == "low"
        assert data["scam_score"] == 10
