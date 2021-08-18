from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Callable,
    Optional,
    Sequence,
    Iterator,
    Mapping,
)

from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
from ml4ht.data.defines import EXCEPTIONS


DataIndex = Mapping[str, Any]
Data = Dict[str, np.ndarray]
# Given `DataIndex`, return (input data, output data)
DataSource = Callable[[DataIndex], Tuple[Data, Data]]
TrainingExample = Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]


def range_epoch_idx_generator(
    true_epoch_len: int,
    shuffle: bool,
    training_epochs_per_true_epoch: int = 1,
) -> Iterator[Sequence[int]]:
    """
    Yields optionally shuffled lists of indices from a range
    :param true_epoch_len: how many indices to yield from
    :param shuffle: Whether to shuffle the sample ids each epoch
    :param training_epochs_per_true_epoch: How many training epochs to break a true epoch into.
        Defaults to 1 to yield true epochs.
    :yield: Epoch worth of sample ids
    """
    assert training_epochs_per_true_epoch <= true_epoch_len
    sample_ids = np.arange(true_epoch_len)
    while True:
        if shuffle:
            np.random.shuffle(sample_ids)
        epochs = np.array_split(sample_ids, training_epochs_per_true_epoch)
        yield from epochs


def sample_id_epoch_generator(
    sample_ids: List[int],
    sample_id_name: str,
    shuffle: bool,
    training_epochs_per_true_epoch: int = 1,
) -> Iterator[Sequence[DataIndex]]:
    """
    Yields optionally shuffled epochs of sample ids
    :param sample_ids: sample ids (e.g. MRNs) for each epoch
    :param sample_id_name: Name of sample_id in output `DataIndex`s
    :param shuffle: Whether to shuffle the sample ids each epoch
    :param training_epochs_per_true_epoch: How many training epochs to break a true epoch into.
        Defaults to 1 to yield true epochs.
    :yield: Epoch worth of sample ids
    """
    epoch_idx_generator = range_epoch_idx_generator(
        true_epoch_len=len(sample_ids),
        shuffle=shuffle,
        training_epochs_per_true_epoch=training_epochs_per_true_epoch,
    )
    for indices in epoch_idx_generator:
        yield [{sample_id_name: sample_ids[idx]} for idx in indices]


class TrainingDataset(IterableDataset):
    # keys in error list
    error_key = "error"
    source_key = "source"
    idx_key = "data_idx"

    def __init__(
        self,
        data_sources: List[DataSource],
        epoch_indices_iterator: Iterator[Sequence[DataIndex]],
        raise_errors: bool = True,
        verbose: bool = True,
    ):
        """

        :param data_sources: Get data for inputs and outputs to model
        :param epoch_indices_iterator: Iterator over epochs of `DataIndex`s
            For example, see `sample_id_epoch_indices` above.
        :param raise_errors: Whether to raise or skip errors during calling of
            `DataFetcher`s
        """
        super(TrainingDataset).__init__()
        self.data_sources = data_sources
        self.epoch_indices_iterator = epoch_indices_iterator
        self.raise_errors = raise_errors
        self.verbose = verbose
        self.errors = []

    def __iter__(self):
        """
        One epoch of training data.
        """
        epoch_indices = next(self.epoch_indices_iterator)
        worker_info = get_worker_info()
        if worker_info is not None:  # multi-process case
            split_indices = np.array_split(epoch_indices, worker_info.num_workers)
            epoch_indices = split_indices[worker_info.id]
        return filter(
            lambda x: x is not None,  # skips the errors
            map(self.training_example, epoch_indices),
        )

    def format_error(self, error: Dict[str, Any]) -> str:
        return f"On index `{error[self.idx_key]}` got error `{error[self.error_key]}` from source `{error[self.source_key]}`"

    def training_example(
        self,
        data_idx: DataIndex,
    ) -> Optional[Tuple[Data, Data]]:
        combined_inputs, combined_outputs = {}, {}
        for data_source in self.data_sources:
            try:
                data_in, data_out = data_source(data_idx)
            except EXCEPTIONS as e:
                if self.raise_errors:
                    raise e
                self.errors.append(
                    {
                        self.error_key: repr(e),
                        self.source_key: data_source.__name__,
                        self.idx_key: data_idx,
                    },
                )
                if self.verbose:
                    print(self.format_error(self.errors[-1]))
                return
            for name, modality in data_in.items():
                if name in combined_inputs:
                    raise ValueError(
                        f"Multiple DataSources share the same input modality called {name}",
                    )
                combined_inputs[name] = modality
            for name, modality in data_out.items():
                if name in combined_outputs:
                    raise ValueError(
                        f"Multiple DataSources share the same output modality called {name}",
                    )
                combined_outputs[name] = modality
        return combined_inputs, combined_outputs
