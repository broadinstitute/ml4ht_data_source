from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml4ht.data.data_description import DataDescription
from ml4ht.data.util.date_selector import (
    NoDatesAvailableError,
    DateRangeOptionPicker,
    first_dt,
    DATE_OPTION_KEY,
)
from ml4ht.data.explore import (
    DATA_DESCRIPTION_COL,
    ERROR_COL,
    NO_LOADING_OPTIONS_ERROR,
    _data_description_summarize_sample_id,
    _pipeline_sample_getter_summarize_sample_id,
    build_df,
    _format_exception,
)
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
    4: {
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
    4: {},
}


TestException = ValueError("Bad sample id.")


class DictionaryDataDescription(DataDescription):
    def __init__(self, data, fail_idx):
        self.data = data
        self.fail_idx = fail_idx

    def get_loading_options(self, sample_id):
        return [{DATE_OPTION_KEY: dt} for dt in self.data[sample_id]]

    def get_raw_data(self, sample_id, loading_option):
        if sample_id == self.fail_idx:
            raise TestException
        dt = loading_option[DATE_OPTION_KEY]
        return self.data[sample_id][dt]

    @property
    def name(self) -> str:
        return f"{super().name}_{self.fail_idx}"


DD1 = DictionaryDataDescription(RAW_DATA_1, 2)
DD2 = DictionaryDataDescription(RAW_DATA_2, -1)


RDS = DateRangeOptionPicker(
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


def simple_summary(sample_id):
    return pd.DataFrame({"sample_id": [sample_id]})


@pytest.mark.parametrize(
    "multiprocess",
    [False, True],
)
def test_build_df(multiprocess):
    expected_df = pd.concat(list(map(simple_summary, [0, 1, 2])))
    df = build_df(simple_summary, [0, 1, 2], multiprocess).sort_values(by="sample_id")
    del df[ERROR_COL]
    assert df.equals(expected_df)


class TestDataDescriptionSummarizeSampleID:
    @pytest.mark.parametrize(
        "data_description",
        [DD1, DD2],
    )
    def test_success(self, data_description):
        df = _data_description_summarize_sample_id(0, data_description)
        for date, value in data_description.data[0].items():
            row = df[df[DATE_OPTION_KEY] == date].iloc[0]
            assert row[DATA_DESCRIPTION_COL] == data_description.name
            assert row["raw_data"] == value

    def test_fail(self):
        data_description = DD1
        df = _data_description_summarize_sample_id(2, data_description)
        for date, value in data_description.data[0].items():
            row = df[df[DATE_OPTION_KEY] == date].iloc[0]
            assert row[DATA_DESCRIPTION_COL] == data_description.name
            assert row[ERROR_COL] == _format_exception(TestException)

    def test_no_loading_options(self):
        data_description = DD2
        row = _data_description_summarize_sample_id(4, data_description).iloc[0]
        assert row[DATA_DESCRIPTION_COL] == data_description.name
        assert row[ERROR_COL] == _format_exception(NO_LOADING_OPTIONS_ERROR)


class TestSampleGetterSummarizeSampleID:
    def test_success(self):
        row = _pipeline_sample_getter_summarize_sample_id(0, PIPE).iloc[0]
        in_batch, out_batch = PIPE(0)
        for name, tensor in {**in_batch, **out_batch}.items():
            assert row[f"{name}_mean"] == tensor.mean()
            assert row[f"{name}_std"] == tensor.std()

    def test_fail_date_select(self):
        row = _pipeline_sample_getter_summarize_sample_id(3, PIPE).iloc[0]
        assert row[ERROR_COL] == _format_exception(NoDatesAvailableError())

    def test_fail_one_data_description(self):
        row = _pipeline_sample_getter_summarize_sample_id(2, PIPE).iloc[0]
        assert row[ERROR_COL] == _format_exception(TestException)

    def test_no_loading_options(self):
        row = _pipeline_sample_getter_summarize_sample_id(4, PIPE).iloc[0]
        assert row[ERROR_COL] == _format_exception(NoDatesAvailableError())
