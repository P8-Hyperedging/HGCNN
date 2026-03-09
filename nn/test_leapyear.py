import pytest
from leapyear import is_leap_year


def test_divisible_by_400_is_leap_year():
    assert is_leap_year(2000) is True


def test_divisible_by_100_not_400_is_not_leap_year():
    assert is_leap_year(1900) is False
    assert is_leap_year(1800) is False


def test_divisible_by_4_not_100_is_leap_year():
    assert is_leap_year(2024) is True
    assert is_leap_year(1996) is True


def test_not_divisible_by_4_is_not_leap_year():
    assert is_leap_year(2023) is False
    assert is_leap_year(1999) is False
