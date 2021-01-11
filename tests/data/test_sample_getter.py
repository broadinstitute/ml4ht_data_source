from datetime import datetime, timedelta

import numpy as np
import pytest

from ml4ht.data.data_description import DataDescription
from ml4ht.data.date_selector import RangeDateSelector, first_dt
from ml4ht.data.sample_getter import PipelineSampleGetter, TensorMap
from ml4ht.data.transformation import Transformation, TransformationType

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
            raise ValueError("Bad sample id.")
        return self.data[sample_id][dt]


DD1 = DictionaryDataDescription(RAW_DATA_1, 2)
DD2 = DictionaryDataDescription(RAW_DATA_2, -1)


def error_on_negative(x, _, __):
    if np.any(x <= 0):
        raise ValueError("Not all values positive.")
    return x


FILTER_ALL_POSITIVE = Transformation(TransformationType.FILTER, error_on_negative)


def multiply_by_state(x, _, state):
    return x * state["factor"]


STATE_MULTIPLY = Transformation(TransformationType.NORMALIZATION, multiply_by_state)


def state_setter_state_multiply(sample_id):
    return {"factor": sample_id}


TMAP_1 = TensorMap(
    "fails_sometimes",
    DD1,
    [FILTER_ALL_POSITIVE],
)
TMAP_2 = TensorMap(
    "never_fails",
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


class TestTensorMap:
    def test_get_tensor(self):
        data = TMAP_1.get_tensor(0, datetime(year=2000, month=3, day=1), None)
        assert data == 10
        data = TMAP_2.get_tensor(0, datetime(year=2000, month=3, day=2), {"factor": -1})
        assert data == -2

    def test_get_tensor_fail_raw_data(self):
        with pytest.raises(ValueError):
            TMAP_1.get_tensor(2, datetime(year=2000, month=3, day=1), None)

    def test_get_tensor_fail_raw_filter(self):
        with pytest.raises(ValueError):
            TMAP_1.get_tensor(1, datetime(year=2000, month=3, day=1), None)

    def test_get_tensor_explore(self):
        tensor_data = TMAP_2.get_tensor_explore(
            0,
            datetime(year=2000, month=3, day=2),
            {"factor": -1},
        )
        assert tensor_data.ok
        assert tensor_data.data.summary == -2
        assert tensor_data.data.dt == datetime(year=2000, month=3, day=2)

    def test_get_tensor_explore_fail_raw_data(self):
        tensor_data = TMAP_1.get_tensor_explore(
            2,
            datetime(year=2000, month=3, day=1),
            None,
        )
        assert not tensor_data.ok
        assert (
            tensor_data.error
            == "Getting raw data failed causing error of type ValueError."
        )

    def test_get_tensor_fail_explore_raw_filter(self):
        tensor_data = TMAP_1.get_tensor_explore(
            1,
            datetime(year=2000, month=3, day=1),
            None,
        )
        assert not tensor_data.ok
        assert (
            tensor_data.error
            == "error_on_negative_TransformationType.FILTER failed causing error of type ValueError."
        )


class TestPipelineSampleGetter:
    def test_call(self):
        in_batch, out_batch = PIPE(0)
        assert in_batch[TMAP_1.input_name] == 10
        # TMAP_2 will be 0 since its transformation multiplies by sample_id
        assert out_batch[TMAP_2.output_name] == 0

    def test_call_fails_raw_data(self):
        with pytest.raises(ValueError):
            PIPE(2)

    def test_call_fails_date_select(self):
        with pytest.raises(ValueError):
            PIPE(3)

    def test_call_fails_filter(self):
        with pytest.raises(ValueError):
            PIPE(1)

    def test_explore_batch(self):
        data = PIPE.explore_batch(0).data
        assert data.state["factor"] == 0

        tensor_data = data.in_batch[TMAP_1.input_name]
        assert tensor_data.ok
        assert tensor_data.data.summary == 10
        assert tensor_data.data.dt == datetime(year=2000, month=3, day=1)

        tensor_data = data.out_batch[TMAP_2.output_name]
        assert tensor_data.ok
        assert tensor_data.data.summary == 0
        assert tensor_data.data.dt == datetime(year=2000, month=3, day=2)

    def test_explore_batch_fails_raw_data(self):
        data = PIPE.explore_batch(2).data
        assert data.state["factor"] == 2

        tensor_data = data.in_batch[TMAP_1.input_name]
        assert not tensor_data.ok
        assert (
            tensor_data.error
            == "Getting raw data failed causing error of type ValueError."
        )

        tensor_data = data.out_batch[TMAP_2.output_name]
        assert tensor_data.ok
        assert tensor_data.data.summary == 20
        assert tensor_data.data.dt == datetime(year=2000, month=3, day=1)

    def test_explore_batch_call_fails_date_select(self):
        result = PIPE.explore_batch(3)
        assert not result.ok
        assert result.error == "Selecting dates failed causing error of type NoDTError."

    def test_explore_batch_fails_filter(self):
        data = PIPE.explore_batch(1).data
        assert data.state["factor"] == 1

        tensor_data = data.in_batch[TMAP_1.input_name]
        assert not tensor_data.ok
        assert (
            tensor_data.error
            == "error_on_negative_TransformationType.FILTER failed causing error of type ValueError."
        )

        tensor_data = data.out_batch[TMAP_2.output_name]
        assert tensor_data.ok
        assert tensor_data.data.summary == 10
        assert tensor_data.data.dt == datetime(year=2000, month=3, day=1)
