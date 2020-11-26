from datetime import datetime
from typing import List, Dict, Any

from ml4h.data.data_description import DataDescription
from ml4h.data.defines import SampleID, Tensor


RAW_DATA = {
    0: {
        datetime(year=2000, month=3, day=1): 10,
        datetime(year=2000, month=3, day=10): -1,
    },
    1: {
        datetime(year=2020, month=12, day=25): 3,
    },
}


class DictionaryDataDescription(DataDescription):
    def __init__(self):
        self.raw_data = RAW_DATA

    def get_dates(self, sample_id: SampleID) -> List[datetime]:
        return list(self.raw_data[sample_id])

    def get_raw_data(self, sample_id: SampleID, dt: datetime) -> Tensor:
        return self.raw_data[sample_id][dt]

    def get_summary_data(self, sample_id: SampleID, dt: datetime) -> Dict[str, Any]:
        summary = super().get_summary_data(sample_id, dt)
        summary['is_positive'] = summary['raw_data'] > 0
        return summary


def test_get_dates():
    ddd = DictionaryDataDescription()
    for sample_id, date_to_value in RAW_DATA.items():
        dates = set(date_to_value.keys())
        assert set(ddd.get_dates(sample_id)) == dates


def test_get_raw_data():
    ddd = DictionaryDataDescription()
    for sample_id, date_to_value in RAW_DATA.items():
        for date, value in date_to_value.items():
            assert ddd.get_raw_data(sample_id, date) == value


def test_get_summary_data():
    ddd = DictionaryDataDescription()
    for sample_id, date_to_value in RAW_DATA.items():
        for date, value in date_to_value.items():
            summary = ddd.get_summary_data(sample_id, date)
            assert summary['raw_data'] == value
            assert summary['is_positive'] == (value > 0)
