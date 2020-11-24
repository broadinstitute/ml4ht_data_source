from typing import Callable, Dict, List

from torch.utils.data import Dataset

from ml4h.data.defines import SampleID, SampleGetter, Batch


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


# TODO: training sample getter, validation, test
