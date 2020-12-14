from datetime import datetime, timedelta

import pytest

from ml4h.data.data_description import DataDescription
from ml4h.data.date_selector import NoDTError, RangeDateSelector, first_dt

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

    def get_dates(self, sample_id):
        return list(self.data[sample_id])

    def get_raw_data(self, sample_id, dt):
        return self.data[sample_id][dt]


DD1 = DictionaryDataDescription(RAW_DATA_1)
DD2 = DictionaryDataDescription(RAW_DATA_2)


def test_select_range_forward():
    rds = RangeDateSelector(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        other_data_descriptions=[DD2],
        time_before=timedelta(days=0),
        time_after=timedelta(days=5),
    )
    dts = rds.select_dates(0)
    assert dts[DD1] == datetime(year=2000, month=3, day=1)
    assert dts[DD2] == datetime(year=2000, month=3, day=2)


def test_select_range_backward():
    rds = RangeDateSelector(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        other_data_descriptions=[DD2],
        time_before=timedelta(days=5),
        time_after=timedelta(days=0),
    )
    dts = rds.select_dates(0)
    assert dts[DD1] == datetime(year=2000, month=3, day=1)
    assert dts[DD2] == datetime(year=2000, month=2, day=29)


def test_select_range_no_dates():
    rds = RangeDateSelector(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        other_data_descriptions=[DD2],
        time_before=timedelta(days=0),
        time_after=timedelta(days=0),
    )
    with pytest.raises(NoDTError):
        rds.select_dates(0)
