import pytest
# This is just a demo of the synthetic data for now


def test_synthetic_data(test_data):
    assert test_data[pytest.CATEGORICAL_NAME].dropna().isin([-1] + list(range(pytest.NUM_CATEGORIES))).all()
