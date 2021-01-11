import numpy as np
from torch.utils.data import DataLoader

from ml4h.data.data_loader import ML4HDataset, numpy_collate_fn
from ml4h.data.defines import Batch


def sample_getter(sample_id: int) -> Batch:
    return {'in': np.full((3, 3), sample_id)}, {'out': np.full((1,), sample_id)}


dataset = ML4HDataset(list(range(100)), sample_getter)


def test_numpy_data_loader():
    """does numpy_collate_fn correctly stack batch data"""
    batch_size = 5
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=numpy_collate_fn)
    for i, (actual_in, actual_out) in enumerate(data_loader):
        for j in range(batch_size):
            expected_in, expected_out = sample_getter(i * batch_size + j)
            for k, v in expected_in.items():
                np.testing.assert_allclose(actual_in[k][j], v)
            for k, v in expected_out.items():
                np.testing.assert_allclose(actual_out[k][j], v)
