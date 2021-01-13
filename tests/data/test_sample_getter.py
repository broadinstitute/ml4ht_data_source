from datetime import datetime, timedelta

import numpy as np
import pytest

from ml4ht.data.data_description import DataDescription
from ml4ht.data.util.date_selector import RangeDateSelector, first_dt, DATE_OPTION_KEY
from ml4ht.data.sample_getter import DataDescriptionSampleGetter

RAW_DATA_1 = {
    0: {
        datetime(year=2000, month=3, day=1): np.array([10]),
    },
    1: {
        datetime(year=2000, month=3, day=1): np.array([-1]),
    },
    2: {
        datetime(year=2000, month=3, day=1): np.array([10]),
    },
    3: {
        datetime(year=2000, month=3, day=1): np.array([10]),
    },
}
RAW_DATA_2 = {
    0: {
        datetime(year=2000, month=3, day=2): np.array([2]),
        datetime(year=2000, month=3, day=3): np.array([3]),
        datetime(year=2000, month=3, day=4): np.array([4]),
        datetime(year=2000, month=2, day=29): np.array([29]),
    },
    1: {
        datetime(year=2000, month=3, day=1): np.array([10]),
    },
    2: {
        datetime(year=2000, month=3, day=1): np.array([10]),
    },
    3: {
        datetime(year=2000, month=3, day=10): np.array([10]),
    },
}


class DictionaryDataDescription(DataDescription):
    def __init__(self, data, fail_idx):
        self.data = data
        self.fail_idx = fail_idx

    def get_loading_options(self, sample_id):
        return [{DATE_OPTION_KEY: dt} for dt in self.data[sample_id]]

    def get_raw_data(self, sample_id, loading_option):
        if sample_id == self.fail_idx:
            raise ValueError("Bad idx")
        dt = loading_option[DATE_OPTION_KEY]
        return self.data[sample_id][dt]


DD1 = DictionaryDataDescription(RAW_DATA_1, 2)
DD2 = DictionaryDataDescription(RAW_DATA_2, -1)


RDS = RangeDateSelector(
    reference_data_description=DD1,
    reference_date_chooser=first_dt,
    time_before=timedelta(days=0),
    time_after=timedelta(days=5),
)
PIPE = DataDescriptionSampleGetter(
    input_data_descriptions=[DD1],
    output_data_descriptions=[DD2],
    option_picker=RDS,
)


class TestPipelineSampleGetter:
    def test_call(self):
        in_batch, out_batch = PIPE(0)
        assert in_batch[DD1.name] == 10
        assert out_batch[DD2.name] == 2

        in_batch, out_batch = PIPE(1)
        assert in_batch[DD1.name] == -1
        assert out_batch[DD2.name] == 10

    def test_call_fails_raw_data(self):
        with pytest.raises(ValueError):
            PIPE(2)

    def test_call_fails_date_select(self):
        with pytest.raises(ValueError):
            PIPE(3)
