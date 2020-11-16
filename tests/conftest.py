import pytest
import datetime
import pandas as pd
import numpy as np


def pytest_configure():
    pytest.N_TENSORS = 50

    # test data
    pytest.TWO_D_DATE_COLUMN = 'date'
    pytest.TWO_D_NAME = '2d_tensor'
    pytest.TWO_D_SHAPE = (5, 1)
    pytest.TWO_D_DATE = datetime.datetime(year=2020, month=1, day=1)
    pytest.TWO_D_FAILURE_IDX = [0, 1, 2]

    pytest.NUM_CATEGORIES = 3
    pytest.CATEGORICAL_DATE_COLUMN = 'dt'
    pytest.CATEGORICAL_NAME = 'categorical_tensor'
    pytest.CATEGORICAL_BEFORE_DATE = pytest.TWO_D_DATE - datetime.timedelta(days=10)
    pytest.CATEGORICAL_BEFORE_FAILURE_IDX = [2, 3, 4]
    pytest.CATEGORICAL_AFTER_DATE = pytest.TWO_D_DATE + datetime.timedelta(days=10)
    pytest.CATEGORICAL_AFTER_FAILURE_IDX = [4, 5, 6]


@pytest.fixture(scope='module')
def test_data():
    data = []
    for i in range(pytest.N_TENSORS):
        category = i % pytest.NUM_CATEGORIES
        two_d = (np.full(pytest.TWO_D_SHAPE, np.nan) if i in pytest.TWO_D_FAILURE_IDX else np.random.randn() + category)
        data.append({
            pytest.TWO_D_DATE_COLUMN: pytest.TWO_D_DATE,
            pytest.TWO_D_NAME: two_d
        })

        cat_before = -1 if i in pytest.CATEGORICAL_BEFORE_FAILURE_IDX else category
        data.append({
            pytest.CATEGORICAL_DATE_COLUMN: pytest.CATEGORICAL_BEFORE_DATE,
            pytest.CATEGORICAL_NAME: cat_before,
        })

        cat_after = -1 if i in pytest.CATEGORICAL_AFTER_FAILURE_IDX else (category + 1) % 3
        data.append({
            pytest.CATEGORICAL_DATE_COLUMN: pytest.CATEGORICAL_AFTER_DATE,
            pytest.CATEGORICAL_NAME: cat_after,
        })

    return pd.DataFrame(data)
