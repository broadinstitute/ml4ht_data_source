from typing import List

import numpy as np
from torch.utils.data import Dataset

from ml4h.data.defines import Batch, SampleGetter, SampleID


class ML4HDataset(Dataset):
    """A pytorch Dataset compatible with ML4H models"""

    def __init__(
        self,
        sample_ids: List[SampleID],
        sample_getter: SampleGetter,
    ):
        self.sample_getter = sample_getter
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, item: int) -> Batch:
        sample_id = self.sample_ids[item]
        return self.sample_getter(sample_id)


def numpy_collate_fn(batches: List[Batch]) -> Batch:
    """
    Merges a list of ml4h batch formatted data.
    Can be used as 'collate_fn` in torch.utils.data.DataLoader
    so that the torch data loader is compatible with tensorflow models
    """
    in_batch = {
        k: np.stack([sample[0][k] for sample in batches]).astype(np.float32)
        for k in batches[0][0]
    }
    out_batch = {
        k: np.stack([sample[1][k] for sample in batches]).astype(np.float32)
        for k in batches[0][1]
    }
    return in_batch, out_batch
