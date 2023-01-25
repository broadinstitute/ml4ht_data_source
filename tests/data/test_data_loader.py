from collections import defaultdict

import pytest
import numpy as np
from torch.utils.data import DataLoader

from ml4ht.data.data_loader import (
    SampleGetterDataset,
    numpy_collate_fn,
    CallbackDataset,
    ML4HCallbackDataset,
    SampleGetterIterableDataset,
    AllLoadingOptionsDataset,
)
from ml4ht.data.defines import Batch
from ml4ht.data.data_description import DataDescription


def sample_getter(sample_id: int) -> Batch:
    return {"in": np.full((3, 3), sample_id)}, {"out": np.full((1,), sample_id)}


ERROR_ID = 3


def error_sample_getter(sample_id: int) -> Batch:
    if sample_id == ERROR_ID:
        raise ValueError()
    return {"in": np.full((3, 3), sample_id)}, {"out": np.full((1,), sample_id)}


sample_getter_dataset = SampleGetterDataset(list(range(100)), sample_getter)


def assert_batches_close(expected: Batch, actual: Batch):
    for expected_half, actual_half in zip(expected, actual):
        for k, v in expected_half.items():
            np.testing.assert_allclose(actual_half[k], v)


def test_numpy_data_loader():
    """does numpy_collate_fn correctly stack batch data"""
    batch_size = 5
    data_loader = DataLoader(
        sample_getter_dataset,
        batch_size=batch_size,
        collate_fn=numpy_collate_fn,
    )
    for i, (actual_in, actual_out) in enumerate(data_loader):
        for j in range(batch_size):
            expected_in, expected_out = sample_getter(i * batch_size + j)
            for k, v in expected_in.items():
                np.testing.assert_allclose(actual_in[k][j], v)
            for k, v in expected_out.items():
                np.testing.assert_allclose(actual_out[k][j], v)


def simple_callback(arg):
    return float(arg)


def simple_transform(tensor):
    return -tensor


def test_callback_dataset():
    dataset = CallbackDataset(
        "floatify",
        callback_args=list(map(str, range(100))),
        callback=simple_callback,
        transforms=[simple_transform],
    )
    assert len(dataset) == 100
    for i, out in enumerate(dataset):
        assert i == -out


def test_ml4h_callback_dataset():
    inp_name = "floatify_in"
    dataset_in = CallbackDataset(
        inp_name,
        callback_args=list(map(str, range(100))),
        callback=simple_callback,
        transforms=[simple_transform],
    )
    out_name = "floatify_out"
    dataset_out = CallbackDataset(
        out_name,
        callback_args=list(map(str, range(100))),
        callback=simple_callback,
    )
    merged = ML4HCallbackDataset([dataset_in], [dataset_out])
    assert len(merged) == 100
    for i, (inp, out) in enumerate(merged):
        assert inp[inp_name] == -i
        assert out[out_name] == i


def test_ml4h_callback_dataset_bad_merge():
    dataset1 = CallbackDataset(
        "floatify",
        callback_args=list(map(str, range(100))),
        callback=simple_callback,
        transforms=[simple_transform],
    )
    dataset2 = CallbackDataset(
        "floatify",
        callback_args=list(map(str, range(101))),
        callback=simple_callback,
        transforms=[simple_transform],
    )
    with pytest.raises(ValueError):
        ML4HCallbackDataset([dataset1], [dataset2])


class TestSampleGetterIterableDataset:
    @pytest.mark.parametrize(
        "multiprocess",
        [False, True],
    )
    @pytest.mark.parametrize(
        [
            SampleGetterIterableDataset.no_shuffle_get_epoch,
            SampleGetterIterableDataset.shuffle_get_epoch,
        ],
    )
    def test_loads_correctly(self, multiprocess, next_epoch):
        sample_ids = list(range(10))
        dataset = SampleGetterIterableDataset(
            sample_ids=sample_ids,
            sample_getter=error_sample_getter,
            get_epoch=next_epoch,
        )
        num_workers = 4 if multiprocess else 0
        loader = DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=numpy_collate_fn,
        )
        found_ids = []
        for batch in loader:
            sample_id = batch[0]["in"][0, 0, 0]
            found_ids.append(sample_id)
            assert_batches_close(numpy_collate_fn([sample_getter(sample_id)]), batch)
        assert len(found_ids) == len(sample_ids) - 1  # right epoch length?
        assert set(found_ids) == set(sample_ids) - {ERROR_ID}  # right epoch contents?

    @pytest.mark.parametrize(
        "multiprocess",
        [False, True],
    )
    @pytest.mark.parametrize(
        [
            SampleGetterIterableDataset.no_shuffle_get_epoch,
            SampleGetterIterableDataset.shuffle_get_epoch,
        ],
    )
    def test_no_working_ids(self, multiprocess, next_epoch):
        sample_ids = [3]
        dataset = SampleGetterIterableDataset(
            sample_ids=sample_ids,
            sample_getter=error_sample_getter,
            get_epoch=next_epoch,
        )
        num_workers = 4 if multiprocess else 0
        loader = DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=numpy_collate_fn,
        )
        with pytest.raises(ValueError):
            list(loader)


class DictionaryDataDescription(DataDescription):
    def __init__(self, name: int):
        self.data = {
            0: {
                0: np.array([2]),
                1: np.array([3]),
                2: np.array([4]),
                3: np.array([29]),
            },
            1: {
                0: np.array([-1]),
            },
            2: {},
            3: {0: np.array([-8])},
        }
        self._name = name

    def get_loading_options(self, sample_id):
        return [{"nice_key": x * self._name} for x in self.data[sample_id]]

    def name(self) -> str:
        return f"dict_dd_{self._name}"

    def get_raw_data(self, sample_id, loading_option):
        dt = loading_option["nice_key"]
        if sample_id == 3:
            raise ValueError("Bad sample id")
        return self.data[sample_id][dt]


def _name_loading_option(sample_id, dds):
    return [{dd: {"nice_key": dd._name} for dd in dds}]


class TestAllLoadingOptionsDataset:
    def test_default_get_loading_options(self):
        dds = [
            DictionaryDataDescription(1),
            DictionaryDataDescription(2),
        ]
        dset = AllLoadingOptionsDataset([0, 1, 2, 3], dds)
        loader = DataLoader(dset, batch_size=2, collate_fn=dset.collate_fn)
        seen_ids = set()
        for model_inputs, sample_ids, loading_options in loader:
            for i, (sample_id, loading_option) in enumerate(
                zip(
                    sample_ids,
                    loading_options,
                ),
            ):
                if sample_id == 2:
                    raise ValueError("Shouldn't see empty sample id")
                if sample_id == 3:
                    raise ValueError("Shouldn't see error-causing sample id")
                for dd in dds:
                    assert np.allclose(
                        model_inputs[dd.name][i],
                        dd.get_raw_data(sample_id, loading_option[dd.name]),
                    )
                seen_ids.add(sample_id)
        assert seen_ids == {0, 1}  # did we see all the ids we wanted to?

    def test_provide_get_loading_options(self):
        dds = [
            DictionaryDataDescription(1),
            DictionaryDataDescription(2),
        ]
        dset = AllLoadingOptionsDataset(
            [0, 1, 2, 3],
            dds,
            _name_loading_option,  # loading option is DD's _name
        )
        loader = DataLoader(dset, batch_size=2, collate_fn=dset.collate_fn)
        seen_ids = set()
        seen_options = defaultdict(set)
        for model_inputs, sample_ids, loading_options in loader:
            for i, (sample_id, loading_option) in enumerate(
                zip(
                    sample_ids,
                    loading_options,
                ),
            ):
                if sample_id == 2:
                    raise ValueError("Shouldn't see empty sample id")
                if sample_id == 3:
                    raise ValueError("Shouldn't see error-causing sample id")
                for dd in dds:
                    option = loading_option[dd.name]
                    assert np.allclose(
                        model_inputs[dd.name][i],
                        dd.get_raw_data(sample_id, option),
                    )
                    seen_options[dd].add(option["nice_key"])
                seen_ids.add(sample_id)
        assert seen_ids == {0}  # did we see all the ids we wanted to?
        for dd in dds:
            assert seen_options[dd] == {dd._name}
