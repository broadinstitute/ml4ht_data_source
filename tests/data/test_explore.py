from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np

from ml4h.data.explore import (
    data_description_summarize_sample_id,
    date_selector_summarize_sample_id,
    pipeline_sample_getter_summarize_sample_id,
    build_df,
    DT_COL,
    ERROR_COL,
    DATA_DESCRIPTION_COL,
    STATE_COL,
)
from ml4h.data.sample_getter import Transformation, TensorMap, TransformationType, PipelineSampleGetter
from ml4h.data.data_description import DataDescription
from ml4h.data.date_selector import RangeDateSelector, first_dt, NoDTError


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

    def get_dates(self, sample_id):
        return list(self.data[sample_id])

    def get_raw_data(self, sample_id, dt):
        if sample_id == self.fail_idx:
            raise ValueError('Bad sample id.')
        return self.data[sample_id][dt]

    @property
    def name(self) -> str:
        return f'{super().name}_{self.fail_idx}'


DD1 = DictionaryDataDescription(RAW_DATA_1, 2)
DD2 = DictionaryDataDescription(RAW_DATA_2, -1)


def error_on_negative(x, _, __):
    if np.any(x <= 0):
        raise ValueError('Not all values positive.')
    return x


FILTER_ALL_POSITIVE = Transformation(TransformationType.FILTER, error_on_negative)


def multiply_by_state(x, _, state):
    return x * state['factor']


STATE_MULTIPLY = Transformation(TransformationType.NORMALIZATION, multiply_by_state)


def state_setter_state_multiply(sample_id):
    return {'factor': sample_id}


TMAP_1 = TensorMap(
    'fails_sometimes',
    DD1,
    [FILTER_ALL_POSITIVE],
)
TMAP_2 = TensorMap(
    'never_fails',
    DD2,
    [STATE_MULTIPLY],
)

RDS = RangeDateSelector(
    reference_data_description=DD1,
    reference_date_chooser=first_dt,
    other_data_descriptions=[DD2],
    time_before=timedelta(days=0),
    time_after=timedelta(days=5),
)
PIPE = PipelineSampleGetter(
    tensor_maps_in=[TMAP_1],
    tensor_maps_out=[TMAP_2],
    date_selector=RDS,
    state_setter=state_setter_state_multiply,
)


def simple_summary(sample_id):
    return pd.DataFrame({'sample_id': [sample_id]})


@pytest.mark.parametrize(
    'multiprocess',
    [False, True]
)
def test_build_df(multiprocess):
    expected_df = pd.concat(list(map(simple_summary, [0, 1, 2])))
    df = build_df(simple_summary, [0, 1, 2], multiprocess).sort_values(by='sample_id')
    assert df.equals(expected_df)


class TestDataDescriptionSummarizeSampleID:

    @pytest.mark.parametrize(
        'data_description',
        [DD1, DD2],
    )
    def test_success(self, data_description):
        df = data_description_summarize_sample_id(0, data_description)
        for date, value in data_description.data[0].items():
            row = df[df[DT_COL] == date].iloc[0]
            assert row[DATA_DESCRIPTION_COL] == data_description.name
            assert row['raw_data'] == value

    def test_fail(self):
        data_description = DD1
        df = data_description_summarize_sample_id(2, data_description)
        for date, value in data_description.data[0].items():
            row = df[df[DT_COL] == date].iloc[0]
            assert row[DATA_DESCRIPTION_COL] == data_description.name
            assert row[ERROR_COL] == 'ValueError'


class TestDateSelectorSummarizeSampleID:

    def test_success(self):
        df = date_selector_summarize_sample_id(0, RDS)
        expected_dates = RDS.select_dates(0)
        for data_selector in (DD1, DD2):
            expected_date = pd.to_datetime(expected_dates[data_selector])
            assert df[f'{DATA_DESCRIPTION_COL}_{data_selector.name}'].iloc[0] == expected_date

    def test_fail(self):
        df = date_selector_summarize_sample_id(3, RDS)
        assert df[ERROR_COL].iloc[0] == NoDTError.__name__


class TestPipelineSampleGetterSummarizeSampeID:

    def test_success(self):
        row = pipeline_sample_getter_summarize_sample_id(0, PIPE).iloc[0]
        expected_in, expected_out, state = PIPE.explore_batch(0).data
        for name, tensor_result in {**expected_in, **expected_out}.items():
            assert row[f'{name}_summary'] == tensor_result.data.summary
            assert row[f'{name}_{DT_COL}'] == tensor_result.data.dt
        assert row[STATE_COL] == state

    def test_fail_date_select(self):
        row = pipeline_sample_getter_summarize_sample_id(3, PIPE).iloc[0]
        assert NoDTError.__name__ in row[ERROR_COL]

    def test_fail_one_tmap(self):
        row = pipeline_sample_getter_summarize_sample_id(2, PIPE).iloc[0]
        expected_in, expected_out, state = PIPE.explore_batch(1).data
        for name, tensor_result in {**expected_in, **expected_out}.items():
            if tensor_result.ok:
                row[f'{name}_summary'] = tensor_result.data.summary
                row[f'{name}_{DT_COL}'] = tensor_result.data.dt
            else:
                assert ValueError.__name__ in row[f'{name}_{ERROR_COL}']
