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
DataFetcher = Callable[[DataIndex], Data]
TrainingExample = Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]


def range_epoch_idx_generator(
    true_epoch_len: int,
    shuffle: bool,
    training_epochs_per_true_epoch: int = 1,
) -> Sequence[int]:
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
) -> List[DataIndex]:
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
    fetcher_key = "data_fetcher"
    idx_key = "data_idx"

    def __init__(
        self,
        input_data_fetchers: List[DataFetcher],
        output_data_fetchers: List[DataFetcher],
        epoch_indices_iterator: Iterator[Sequence[DataIndex]],
        raise_errors: bool = True,
    ):
        """

        :param input_data_fetchers: Get data for inputs to model
        :param output_data_fetchers: Get data for outputs of model
        :param epoch_indices_iterator: Iterator over epochs of `DataIndex`s
            For example, see `sample_id_epoch_indices` above.
        :param raise_errors: Whether to raise or skip errors during calling of
            `DataFetcher`s
        """
        super(TrainingDataset).__init__()
        self.input_data_fetchers = input_data_fetchers
        self.output_data_fetchers = output_data_fetchers
        self.epoch_indices_iterator = epoch_indices_iterator
        self.raise_errors = raise_errors
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

    def _half_example(
        self,
        data_idx: DataIndex,
        is_input: bool,
    ) -> Tuple[Dict[str, np.ndarray], bool]:
        data_fetchers = (
            self.input_data_fetchers if is_input else self.output_data_fetchers
        )
        out = {}
        for data_fetcher in data_fetchers:
            try:
                fetched_data = data_fetcher(data_idx)
            except EXCEPTIONS as e:
                if self.raise_errors:
                    raise e
                self.errors.append(
                    {
                        self.error_key: repr(e),
                        self.fetcher_key: data_fetcher.__name__,
                        self.idx_key: data_idx,
                    },
                )
                return {}, True
            for name, modality in fetched_data.items():
                if name in out:
                    raise ValueError(
                        f"Multiple DataFetchers share the same modality called {name}",
                    )
                out[name] = modality
        return out, False

    def training_example(self, data_idx: DataIndex) -> Optional[TrainingExample]:
        example_in, got_error = self._half_example(data_idx, True)
        if got_error:
            return
        example_out, got_error = self._half_example(data_idx, False)
        if got_error:
            return
        return example_in, example_out
