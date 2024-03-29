from datetime import datetime, timedelta

import pytest

from ml4ht.data.data_description import DataDescription
from ml4ht.data.util.date_selector import (
    NoDatesAvailableError,
    DateRangeOptionPicker,
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


SAMPLE_ID_KEY = "sample_id"


class DictionaryDataDescription(DataDescription):
    def __init__(self, data):
        self.data = data

    def get_loading_options(self, sample_id):
        return [
            {
                DATE_OPTION_KEY: dt,
                SAMPLE_ID_KEY: sample_id,
            }
            for dt in self.data[sample_id]
        ]

    def get_raw_data(self, sample_id, loading_option):
        dt = loading_option[DATE_OPTION_KEY]
        return self.data[sample_id][dt]


DD1 = DictionaryDataDescription(RAW_DATA_1)
DD2 = DictionaryDataDescription(RAW_DATA_2)


def test_select_range_forward():
    rds = DateRangeOptionPicker(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        time_before=timedelta(days=0),
        time_after=timedelta(days=5),
    )
    dts = rds(0, [DD1, DD2])
    assert dts[DD1][DATE_OPTION_KEY] == datetime(year=2000, month=3, day=1)
    assert dts[DD1][SAMPLE_ID_KEY] == 0
    assert dts[DD2][DATE_OPTION_KEY] == datetime(year=2000, month=3, day=2)
    assert dts[DD2][SAMPLE_ID_KEY] == 0


def test_select_range_backward():
    rds = DateRangeOptionPicker(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        time_before=timedelta(days=5),
        time_after=timedelta(days=0),
    )
    dts = rds(0, [DD1, DD2])
    assert dts[DD1][DATE_OPTION_KEY] == datetime(year=2000, month=3, day=1)
    assert dts[DD1][SAMPLE_ID_KEY] == 0
    assert dts[DD2][DATE_OPTION_KEY] == datetime(year=2000, month=2, day=29)
    assert dts[DD2][SAMPLE_ID_KEY] == 0


def test_select_range_no_dates():
    rds = DateRangeOptionPicker(
        reference_data_description=DD1,
        reference_date_chooser=first_dt,
        time_before=timedelta(days=0),
        time_after=timedelta(days=0),
    )
    with pytest.raises(NoDatesAvailableError):
        rds(0, [DD1, DD2])
