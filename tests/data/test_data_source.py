import pytest
import numpy as np
from torch.utils.data import DataLoader
from ml4ht.data.data_source import (
    range_epoch_idx_generator,
    sample_id_epoch_generator,
    TrainingDataset,
    DataIndex,
)
from ml4ht.data.data_loader import numpy_collate_fn


class TestRangeEpochIdxGenerator:
    def test_no_shuffle_true_epoch(self):
        epoch_len = 100
        expected_epoch = np.arange(epoch_len)
        idx_generator = range_epoch_idx_generator(
            true_epoch_len=epoch_len,
            shuffle=False,
            training_epochs_per_true_epoch=1,
        )
        for _ in range(3):
            epoch_indices = next(idx_generator)
            assert (epoch_indices == expected_epoch).all()

    def test_shuffle_true_epoch(self):
        epoch_len = 100
        expected_epoch = np.arange(epoch_len)
        idx_generator = range_epoch_idx_generator(
            true_epoch_len=epoch_len,
            shuffle=True,
            training_epochs_per_true_epoch=1,
        )
        for _ in range(3):
            epoch_indices = next(idx_generator)
            sorted_indices = np.sort(epoch_indices)
            # did we see the all expected idxs
            assert (sorted_indices == expected_epoch).all()
            # were they shuffled?
            assert (epoch_indices != expected_epoch).any()

    def test_no_shuffle_smaller_epoch(self):
        true_epoch_len = 100
        training_steps_per_epoch = 5
        idx_generator = range_epoch_idx_generator(
            true_epoch_len=true_epoch_len,
            shuffle=False,
            training_epochs_per_true_epoch=training_steps_per_epoch,
        )
        epoch_len = true_epoch_len // training_steps_per_epoch
        for i in range(training_steps_per_epoch * 2):
            epoch_indices = next(idx_generator)
            start_idx = (i % training_steps_per_epoch) * epoch_len
            stop_idx = start_idx + epoch_len
            assert (epoch_indices == np.arange(start_idx, stop_idx)).all()

    def test_shuffle_smaller_epoch(self):
        true_epoch_len = 100
        training_steps_per_epoch = 6
        idx_generator = range_epoch_idx_generator(
            true_epoch_len=true_epoch_len,
            shuffle=True,
            training_epochs_per_true_epoch=training_steps_per_epoch,
        )
        for _ in range(2):
            seen_ids = []
            for _ in range(training_steps_per_epoch):
                seen_ids += list(next(idx_generator))
            # did we see the all expected idxs
            assert sorted(seen_ids) == list(range(true_epoch_len))
            # were they shuffled?
            assert seen_ids != list(range(true_epoch_len))


def _input_data_source(idx: DataIndex):
    i = np.array(idx["sample_id"])
    if i == 5:
        raise ValueError
    return {"a": i, "b": -i}, {}


def _output_data_source(idx: DataIndex):
    i = np.array(idx["sample_id"])
    return {}, {"c": i, "d": -i}


def assert_batches_close(expected, actual):
    for expected_half, actual_half in zip(expected, actual):
        for k, v in expected_half.items():
            np.testing.assert_allclose(actual_half[k], v)


class TestTrainingDataset:
    def test_skip_errors(self):
        epoch_generator = sample_id_epoch_generator(
            list(range(6)),
            "sample_id",
            True,
            1,
        )
        dset = TrainingDataset(
            [_input_data_source, _output_data_source],
            epoch_generator,
            raise_errors=False,
            verbose=True,
        )
        inputs = []
        outputs = []
        for x, y in iter(dset):
            inputs.append(x)
            outputs.append(y)
        for i in range(5):
            assert _input_data_source({"sample_id": i})[0] in inputs
            assert _output_data_source({"sample_id": i})[1] in outputs
        assert dset.errors[0] == {
            dset.error_key: repr(ValueError()),
            dset.idx_key: {"sample_id": 5},
            dset.source_key: _input_data_source.__name__,
        }

    def test_raise_errors(self):
        epoch_generator = sample_id_epoch_generator(
            [5],
            "sample_id",
            True,
            1,
        )
        dset = TrainingDataset(
            [_input_data_source, _output_data_source],
            epoch_generator,
            raise_errors=True,
        )
        with pytest.raises(ValueError):
            list(iter(dset))

    def test_overlapping_inputs(self):
        epoch_generator = sample_id_epoch_generator(
            [0],
            "sample_id",
            True,
            1,
        )
        dset = TrainingDataset(
            [_input_data_source, _input_data_source],
            epoch_generator,
            raise_errors=True,
        )
        with pytest.raises(ValueError):
            list(iter(dset))

    @pytest.mark.parametrize(
        "multiprocess",
        [False, True],
    )
    def test_works_in_loader(self, multiprocess):
        epoch_generator = sample_id_epoch_generator(
            list(range(6)),
            "sample_id",
            True,
            1,
        )
        dset = TrainingDataset(
            [_input_data_source, _output_data_source],
            epoch_generator,
            raise_errors=False,
        )
        loader = DataLoader(
            dset,
            batch_size=2,
            collate_fn=numpy_collate_fn,
            num_workers=2 if multiprocess else 0,
            drop_last=False,
        )
        found_ids = []
        for batch in loader:
            sample_ids = batch[0]["a"]
            found_ids += list(sample_ids)
            assert_batches_close(
                numpy_collate_fn(
                    [
                        dset.training_example({"sample_id": sample_id})
                        for sample_id in sample_ids
                    ],
                ),
                batch,
            )
        assert len(found_ids) == 6 - 1  # right epoch length?
        assert set(found_ids) == set(range(5))  # right epoch contents?
