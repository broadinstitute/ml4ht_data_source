from typing import List, Callable, TypeVar

import numpy as np
from torch.utils.data import Dataset

from ml4ht.data.defines import Batch, SampleGetter, SampleID, Tensor


class SampleGetterDataset(Dataset):
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


CallbackArg = TypeVar("CallbackArg")


class CallbackDataset(Dataset):
    """
    A dataset representing a single modality.
    Designed to be merged with other datasets
    to fit the ML4H format
    """

    def __init__(
        self,
        name: str,
        callback: Callable[[CallbackArg], Tensor],
        callback_args: List[CallbackArg],
        transforms: List[Callable[[Tensor], Tensor]] = None,
    ):
        self._name = name
        self.callback = callback
        self.callback_args = callback_args
        self.transforms = transforms or []

    def __len__(self) -> int:
        return len(self.callback_args)

    def __getitem__(self, item: int) -> Tensor:
        tensor = self.callback(self.callback_args[item])
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor

    @property
    def name(self) -> str:
        return self._name


class ML4HCallbackDataset(Dataset):
    """
    Merges CallbackDatasets into a single dataset, which puts data in the ML4H batch format.
    You must be __sure__ that all of the datasets' callback_args are in the same order.
    """

    def __init__(
        self,
        input_callback_datasets: List[CallbackDataset],
        output_callback_datasets: List[CallbackDataset],
    ):
        self.input_callback_datasets = input_callback_datasets
        self.output_callback_datasets = output_callback_datasets
        self.length = self._validate_datasets_mergeable(
            self.input_callback_datasets + self.output_callback_datasets,
        )

    @staticmethod
    def _validate_datasets_mergeable(datasets: List[CallbackDataset]) -> int:
        """All the merged datasets must be the same length"""
        lengths = set(len(dataset) for dataset in datasets)
        if len(lengths) != 1:
            raise ValueError("You cannot merge CallbackDatasets of different lengths.")
        return lengths.pop()

    def __len__(self):
        return self.length

    def __getitem__(self, item: int) -> Batch:
        return (
            {
                callback.name: callback[item]
                for callback in self.input_callback_datasets
            },
            {
                callback.name: callback[item]
                for callback in self.output_callback_datasets
            },
        )


def numpy_collate_fn(batches: List[Batch]) -> Batch:
    """
    Merges a list of ml4ht batch formatted data.
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
