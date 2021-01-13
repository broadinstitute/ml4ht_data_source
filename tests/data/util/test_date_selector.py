from datetime import datetime, timedelta

import pytest

from ml4ht.data.data_description import DataDescription
from ml4ht.data.util.date_selector import (
    NoDTError,
    RangeDateSelector,
    first_dt,
    DATE_OPTION_KEY,
)

RAW_DATA_1 = {
    0: {
        datetime(year=2000, month=3, day=1): 10,
    },
}
RAW_DATA_2 = {
    0: {
        datetime(year=2000, month=3, day=2): 2,
        datetime(year=2000, month=3, day=3): 3,
        datetime(year=2000, month=3, day=4): 4,
        datetime(year=2000, month=2, day=29): 29,
    },
}


class DictionaryDataDescription(DataDescription):
    def __init__(self, data):
        self.data = data

    def get_loading_options(self, sample_id):
        return [{DATE_OPTION_KEY: dt} for dt in self.data[sample_id]]

    def get_raw_data(self, sample_id, loading_option):
        dt = loading_option[DATE_OPTION_KEY]
        return self.data[sample_id][dt]


DD1 = DictionaryDataDescription(RAW_DATA_1)
DD2 = DictionaryDataDescription(RAW_DATA_2)


def test_select_range_forward():
    rds = RangeDateSelector(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        time_before=timedelta(days=0),
        time_after=timedelta(days=5),
    )
    dts = rds(0, [DD1, DD2])
    assert dts[DD1][DATE_OPTION_KEY] == datetime(year=2000, month=3, day=1)
    assert dts[DD2][DATE_OPTION_KEY] == datetime(year=2000, month=3, day=2)


def test_select_range_backward():
    rds = RangeDateSelector(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        time_before=timedelta(days=5),
        time_after=timedelta(days=0),
    )
    dts = rds(0, [DD1, DD2])
    assert dts[DD1][DATE_OPTION_KEY] == datetime(year=2000, month=3, day=1)
    assert dts[DD2][DATE_OPTION_KEY] == datetime(year=2000, month=2, day=29)


def test_select_range_no_dates():
    rds = RangeDateSelector(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        time_before=timedelta(days=0),
        time_after=timedelta(days=0),
    )
    with pytest.raises(NoDTError):
        rds(0, [DD1, DD2])
