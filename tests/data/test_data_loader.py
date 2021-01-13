import pytest
import numpy as np
from torch.utils.data import DataLoader

from ml4ht.data.data_loader import (
    SampleGetterDataset,
    numpy_collate_fn,
    CallbackDataset,
    ML4HCallbackDataset,
)
from ml4ht.data.defines import Batch


def sample_getter(sample_id: int) -> Batch:
    return {"in": np.full((3, 3), sample_id)}, {"out": np.full((1,), sample_id)}


sample_getter_dataset = SampleGetterDataset(list(range(100)), sample_getter)


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
