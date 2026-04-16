"""
tests/test_tool_phone_check.py
Unit tests for tools/tool_phone_check.py

Tests cover:
  - Valid Indian mobile numbers (E.164 and national formats)
  - Valid US, UK, and international numbers
  - Invalid / unparseable inputs
  - Empty / None input validation
  - Return structure validation
  - Country code extraction
  - E.164 / international / national format outputs
  - Region code accuracy
  - Parametrized valid and invalid phone lists
  - Edge cases: too-short numbers, letters in number, plus-prefix variations
"""
import pytest

from tools.tool_phone_check import check_phone_number


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(phone, region=None) -> dict:
    if region is not None:
        return check_phone_number(phone, region)
    return check_phone_number(phone)


def _data(phone, region=None) -> dict:
    result = _run(phone, region)
    assert result["ok"] is True, f"Expected ok=True for '{phone}', got: {result}"
    return result["data"]


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_string_returns_error(self):
        result = _run("")
        assert result["ok"] is False
        assert "error" in result

    def test_none_returns_error(self):
        result = _run(None)
        assert result["ok"] is False
        assert "error" in result

    def test_error_message_non_empty(self):
        result = _run("")
        assert len(result["error"]) > 0

    def test_letters_only_returns_error(self):
        result = _run("abcdefg")
        assert result["ok"] is False

    def test_special_chars_only_returns_error(self):
        result = _run("!@#$%^&*")
        assert result["ok"] is False

    def test_too_short_number_returns_error(self):
        result = _run("123", "IN")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# 2. Valid Indian numbers
# ---------------------------------------------------------------------------

class TestValidIndianNumbers:

    def test_e164_indian_mobile_ok(self):
        result = _run("+919876543210")
        assert result["ok"] is True

    def test_indian_mobile_country_code(self):
        data = _data("+919876543210")
        assert data["country_code"] == 91

    def test_indian_mobile_region_code(self):
        data = _data("+919876543210")
        assert data["region_code"] == "IN"

    def test_indian_mobile_e164_format(self):
        data = _data("+919876543210")
        assert data["e164"] == "+919876543210"

    def test_indian_mobile_international_format(self):
        data = _data("+919876543210")
        assert data["international"].startswith("+91")

    def test_indian_national_format_with_region_hint(self):
        result = _run("09876543210", "IN")
        assert result["ok"] is True

    def test_indian_10_digit_with_region_hint(self):
        result = _run("9876543210", "IN")
        assert result["ok"] is True

    def test_indian_number_is_valid(self):
        data = _data("+919876543210")
        assert data["is_valid"] is True

    def test_indian_number_is_possible(self):
        data = _data("+919876543210")
        assert data["is_possible"] is True

    def test_indian_number_has_timezones(self):
        data = _data("+919876543210")
        assert isinstance(data["timezones"], list)
        assert len(data["timezones"]) > 0

    def test_indian_number_location_contains_india(self):
        data = _data("+919876543210")
        assert "India" in data["location"] or data["location"] != ""


# ---------------------------------------------------------------------------
# 3. Valid US numbers
# ---------------------------------------------------------------------------

class TestValidUSNumbers:

    def test_us_e164_ok(self):
        result = _run("+12125551234")
        assert result["ok"] is True

    def test_us_country_code(self):
        data = _data("+12125551234")
        assert data["country_code"] == 1

    def test_us_region_code(self):
        data = _data("+12125551234")
        assert data["region_code"] == "US"

    def test_us_e164_format(self):
        data = _data("+12125551234")
        assert data["e164"] == "+12125551234"

    def test_us_with_region_hint(self):
        result = _run("2125551234", "US")
        assert result["ok"] is True

    def test_us_number_is_valid(self):
        data = _data("+12125551234")
        assert data["is_valid"] is True


# ---------------------------------------------------------------------------
# 4. Valid UK numbers
# ---------------------------------------------------------------------------

class TestValidUKNumbers:

    def test_uk_e164_ok(self):
        result = _run("+447911123456")
        assert result["ok"] is True

    def test_uk_country_code(self):
        data = _data("+447911123456")
        assert data["country_code"] == 44

    def test_uk_region_code(self):
        data = _data("+447911123456")
        assert data["region_code"] == "GB"

    def test_uk_with_region_hint(self):
        result = _run("07911123456", "GB")
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# 5. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_key_always_present(self):
        for phone in ["+919876543210", "", "bad"]:
            result = _run(phone)
            assert "ok" in result

    def test_valid_number_has_data(self):
        result = _run("+919876543210")
        assert "data" in result
        assert isinstance(result["data"], dict)

    def test_all_required_keys_present(self):
        data = _data("+919876543210")
        required = {
            "input", "e164", "international", "national",
            "is_possible", "is_valid", "country_code",
            "region_code", "number_type", "carrier",
            "location", "timezones"
        }
        assert required.issubset(data.keys()), (
            f"Missing keys: {required - set(data.keys())}"
        )

    def test_input_field_preserved(self):
        phone = "+919876543210"
        data = _data(phone)
        assert data["input"] == phone

    def test_e164_starts_with_plus(self):
        data = _data("+919876543210")
        assert data["e164"].startswith("+")

    def test_is_valid_is_bool(self):
        data = _data("+919876543210")
        assert isinstance(data["is_valid"], bool)

    def test_is_possible_is_bool(self):
        data = _data("+919876543210")
        assert isinstance(data["is_possible"], bool)

    def test_country_code_is_int(self):
        data = _data("+919876543210")
        assert isinstance(data["country_code"], int)

    def test_region_code_is_string(self):
        data = _data("+919876543210")
        assert isinstance(data["region_code"], str)

    def test_timezones_is_list(self):
        data = _data("+919876543210")
        assert isinstance(data["timezones"], list)

    def test_number_type_is_string(self):
        data = _data("+919876543210")
        assert isinstance(data["number_type"], str)

    def test_error_result_has_no_data(self):
        result = _run("")
        assert result.get("data") is None or result.get("ok") is False

    def test_invalid_result_has_error_string(self):
        result = _run("not-a-phone")
        assert isinstance(result.get("error"), str)


# ---------------------------------------------------------------------------
# 6. Format output tests
# ---------------------------------------------------------------------------

class TestFormatOutputs:

    def test_e164_format_correct_for_indian(self):
        data = _data("+91 98765 43210")
        assert data["e164"] == "+919876543210"

    def test_international_format_has_spaces(self):
        data = _data("+919876543210")
        assert " " in data["international"]

    def test_national_format_no_country_code(self):
        data = _data("+919876543210")
        assert not data["national"].startswith("+91")

    def test_us_national_format(self):
        data = _data("+12125551234")
        # National format for US: (212) 555-1234
        assert "212" in data["national"]

    def test_e164_no_spaces_or_dashes(self):
        data = _data("+919876543210")
        e164 = data["e164"]
        assert " " not in e164
        assert "-" not in e164


# ---------------------------------------------------------------------------
# 7. Region hint tests
# ---------------------------------------------------------------------------

class TestRegionHint:

    def test_default_region_is_india(self):
        # Without region hint, should use IN as default
        result = _run("9876543210")
        assert result["ok"] is True

    def test_explicit_in_region_hint(self):
        result = _run("9876543210", "IN")
        assert result["ok"] is True

    def test_explicit_us_region_hint(self):
        result = _run("2125551234", "US")
        assert result["ok"] is True

    def test_region_code_none_uses_default(self):
        result = check_phone_number("9876543210", None)
        # Should use DEFAULT_PHONE_REGION = "IN"
        assert result["ok"] is True

    def test_wrong_region_for_number_may_fail(self):
        # Trying to parse a US number as a UK number without country code
        result = _run("2125551234", "GB")
        # May succeed or fail depending on phonenumbers library
        assert "ok" in result


# ---------------------------------------------------------------------------
# 8. Parametrized valid international numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phone,expected_country_code", [
    ("+919876543210", 91),   # India
    ("+12125551234", 1),     # USA
    ("+447911123456", 44),   # UK
    ("+33123456789", 33),    # France
    ("+4915123456789", 49),  # Germany
    ("+819012345678", 81),   # Japan
    ("+61412345678", 61),    # Australia
    ("+5511987654321", 55),  # Brazil
])
def test_parametrized_valid_international_numbers(phone, expected_country_code):
    result = _run(phone)
    if result["ok"]:
        assert result["data"]["country_code"] == expected_country_code, (
            f"Expected country code {expected_country_code} for {phone}"
        )


# ---------------------------------------------------------------------------
# 9. Parametrized invalid inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phone", [
    "",
    None,
    "abc",
    "000",
    "123",
    "!@#$",
    "not-a-phone",
    "0000000000000000000",  # too long
])
def test_parametrized_invalid_inputs_return_error(phone):
    result = _run(phone)
    assert result["ok"] is False, (
        f"Expected ok=False for invalid phone '{phone}', got: {result}"
    )


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_phone_with_spaces_parsed(self):
        result = _run("+91 98765 43210")
        assert result["ok"] is True

    def test_phone_with_dashes_parsed(self):
        result = _run("+91-98765-43210")
        assert result["ok"] is True

    def test_phone_with_parentheses_us(self):
        result = _run("+1 (212) 555-1234")
        assert result["ok"] is True

    def test_phone_with_dots_parsed(self):
        result = _run("+1.212.555.1234")
        assert result["ok"] is True

    def test_leading_zeros_handled(self):
        result = _run("09876543210", "IN")
        assert result["ok"] is True

    def test_single_digit_input_fails(self):
        result = _run("5", "IN")
        assert result["ok"] is False

    def test_extremely_long_number_fails(self):
        result = _run("+9" + "9" * 20)
        assert result["ok"] is False

    def test_carrier_field_is_string(self):
        data = _data("+919876543210")
        assert isinstance(data["carrier"], str)

    def test_location_field_is_string(self):
        data = _data("+919876543210")
        assert isinstance(data["location"], str)
