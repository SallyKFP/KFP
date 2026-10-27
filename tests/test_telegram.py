import pytest

from core.telegram import parse_allowed_user_ids


def test_parse_allowed_user_ids():
    assert parse_allowed_user_ids("123, 456") == {123, 456}


def test_parse_allowed_user_ids_requires_numeric_id():
    with pytest.raises(ValueError, match="numeric IDs"):
        parse_allowed_user_ids("not-an-id")


def test_parse_allowed_user_ids_requires_at_least_one_id():
    with pytest.raises(ValueError, match="Set TELEGRAM_ALLOWED_USER_IDS"):
        parse_allowed_user_ids("")
