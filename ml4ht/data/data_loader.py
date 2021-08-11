from typing import List, Callable, TypeVar, Dict, Any

import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ml4ht.data.defines import (
    Batch,
    SampleGetter,
    SampleID,
    Tensor,
    EXCEPTIONS,
    LoadingOption,
)
from ml4ht.data.data_description import DataDescription
from ml4ht.data.explore import explore_sample_getter


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
        self.true_epochs = 0

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
            self.true_epochs += 1
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            print(
                f"Worker {worker_id}: Epoch completed with {successful_batches} / {len(sample_ids)} successful samples, {self.true_epochs} epochs completed.",
            )
        else:
            explore_df = explore_sample_getter(self.sample_getter, sample_ids)
            raise ValueError(
                f"Visited all {len(sample_ids)} sample ids without finding any valid samples.\n\nErrors:\n{explore_df['error'].value_counts()}",
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
        data_descriptions: List[DataDescription],
        get_loading_options: Callable[
            [SampleID, List[DataDescription]],
            List[LoadingOption],
        ] = None,
        raise_errors: bool = False,
    ):
        super(SampleGetterIterableDataset).__init__()
        self.dds = data_descriptions
        self.sample_ids = sample_ids
        self.get_loading_options: Callable[
            [SampleID, List[DataDescription]],
            List[LoadingOption],
        ] = (
            get_loading_options or self._default_get_loading_options
        )
        self.raise_errors = raise_errors

    def _default_get_loading_options(
        self,
        sample_id: SampleID,
        _,
    ) -> List[Dict[DataDescription, LoadingOption]]:
        """
        Get the loading option for all data descriptions from the first DataDescription
        """
        loading_options = self.dds[0].get_loading_options(sample_id)
        return [
            {dd: loading_option for dd in self.dds}
            for loading_option in loading_options
        ]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:  # multi-process case
            split_sample_ids = np.array_split(self.sample_ids, worker_info.num_workers)
            self.sample_ids = split_sample_ids[worker_info.id]
        return iter(self.one_epoch())

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _one_sample_id(self, sample_id: SampleID):
        for loading_option in self.get_loading_options(sample_id, self.dds):
            try:
                model_input = {
                    dd.name: dd.get_raw_data(sample_id, loading_option[dd])
                    for dd in self.dds
                }
                loading_option_str_key = {
                    dd.name: option  # use names rather than objects for keys
                    for dd, option in loading_option.items()
                }
                yield (
                    sample_id,
                    loading_option_str_key,
                    model_input,
                )
            except EXCEPTIONS as e:
                if self.raise_errors:
                    raise e
                print(
                    f"Got error {repr(e)} on sample id {sample_id} with loading option {loading_option}.",
                )

    def one_epoch(self) -> Batch:
        for sample_id in self.sample_ids:
            yield from self._one_sample_id(sample_id)

    @staticmethod
    def collate_fn(samples):
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

        first_input = samples[0][2]
        in_batch_keys = list(first_input)
        in_batch = {
            k: np.empty((len(samples),) + first_input[k].shape, dtype=np.float32)
            for k in in_batch_keys
        }

        for i, (sample_id, loading_option, model_input) in enumerate(samples):
            sample_ids.append(sample_id)
            loading_options.append(loading_option)
            for k in in_batch_keys:
                in_batch[k][i] = model_input[k]
        return in_batch, sample_ids, loading_options


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


def numpy_collate_fn(samples: List[Batch]) -> Batch:
    """
    Merges a list of ml4ht batch formatted data.
    Can be used as 'collate_fn` in torch.utils.data.DataLoader
    so that the torch data loader is compatible with tensorflow models
    """
    # construct correctly-shaped empty arrays for input and output of model
    in_batch_keys = list(samples[0][0])
    in_batch = {
        k: np.empty((len(samples),) + samples[0][0][k].shape, dtype=np.float32)
        for k in in_batch_keys
    }
    out_batch_keys = list(samples[0][1])
    out_batch = {
        k: np.empty((len(samples),) + samples[0][1][k].shape, dtype=np.float32)
        for k in out_batch_keys
    }
    # fill in the values of the input and output arrays
    for i, sample in enumerate(samples):
        for k in in_batch_keys:
            in_batch[k][i] = sample[0][k]
        for k in out_batch_keys:
            out_batch[k][i] = sample[1][k]
    return in_batch, out_batch
