from typing import List, Callable, TypeVar, Dict, Any

import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ml4ht.data.defines import Batch, SampleGetter, SampleID, Tensor, EXCEPTIONS
from ml4ht.data.data_description import DataDescription


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


class SampleGetterIterableDataset(IterableDataset):
    """
    A pytorch Dataset compatible with ML4H models that gracefully skips errors.
    Uses a SampleGetter, a list of sample ids, and a function to pick the order of the sample ids.
    """

    def __init__(
        self,
        sample_ids: List[SampleID],
        sample_getter: SampleGetter,
        get_epoch: Callable[[List[SampleID]], List[SampleID]] = None,
    ):
        super(SampleGetterIterableDataset).__init__()
        self.sample_getter = sample_getter
        self.sample_ids = sample_ids
        self.get_epoch = get_epoch or self.no_shuffle_get_epoch

    @staticmethod
    def no_shuffle_get_epoch(sample_ids: List[SampleID]) -> List[SampleID]:
        """Non-shuffling epoch"""
        return sample_ids

    @staticmethod
    def shuffle_get_epoch(sample_ids: List[SampleID]) -> List[SampleID]:
        """Shuffling epoch"""
        return list(np.random.permutation(sample_ids))

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:  # multi-process case
            split_sample_ids = np.array_split(self.sample_ids, worker_info.num_workers)
            self.sample_ids = split_sample_ids[worker_info.id]
        return iter(self.one_epoch())

    def __len__(self) -> int:
        return len(self.sample_ids)

    def one_epoch(self) -> Batch:
        sample_ids = self.get_epoch(self.sample_ids)
        successful_batches = 0
        for sample_id in sample_ids:
            try:
                batch = self.sample_getter(sample_id)
                successful_batches += 1
                yield batch
            except EXCEPTIONS as e:
                continue
        if successful_batches:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            print(
                f"Worker {worker_id}: Epoch completed with {successful_batches} / {len(sample_ids)} successful samples",
            )
        else:
            raise ValueError(
                f"Visited all {len(sample_ids)} sample ids without finding any valid samples.",
            )


class AllLoadingOptionsDataset(IterableDataset):
    """
    A pytorch Dataset used for fast inference on a single DataDescription.
    It iterates through every loading option for the DataDescription.
    It keeps track of loading options and sample ids for you.
    It skips errors automatically.
    It should be used in a torch data loader with its collate function.
    """

    def __init__(
        self,
        sample_ids: List[SampleID],
        data_description: DataDescription,
    ):
        super(SampleGetterIterableDataset).__init__()
        self.dd = data_description
        self.sample_ids = sample_ids

        # TODO: remove these edge cases
        assert self.dd.name not in {"loading_option", "sample_id"}

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:  # multi-process case
            split_sample_ids = np.array_split(self.sample_ids, worker_info.num_workers)
            self.sample_ids = split_sample_ids[worker_info.id]
        return iter(self.one_epoch())

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _one_sample_id(self, sample_id: SampleID):
        for loading_option in self.dd.get_loading_options(sample_id):
            try:
                yield {
                    "model_input": self.dd.get_raw_data(sample_id, loading_option),
                    "loading_option": loading_option,
                    "sample_id": sample_id,
                }
            except EXCEPTIONS as e:
                print(
                    f"Got error {repr(e)} on sample id {sample_id} with loading option {loading_option}.",
                )

    def one_epoch(self) -> Batch:
        for sample_id in self.sample_ids:
            yield from self._one_sample_id(sample_id)

    @staticmethod
    def collate_fn(samples: List[Dict[str, Any]]):
        """
        The result of using this collate function with a torch data loader can be used as follows:

            dset = AllLoadingOptionsDataset(...)
            loader = data_loader(dset, collate_fn=AllLoadingOptionsDataset.collate_fn, batch_size=...)
            for model_input, sample_ids, loading_options in loader:
                pred = model.predict(model_input)
                ...
        """
        sample_ids = []
        loading_options = []
        model_inputs = []
        for sample in samples:
            sample_ids.append(sample["sample_id"])
            loading_options.append(sample["loading_option"])
            model_inputs.append(sample["model_input"].astype(np.float32))
        return np.stack(model_inputs), sample_ids, loading_options


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
