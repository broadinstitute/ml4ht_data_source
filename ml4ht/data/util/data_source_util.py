from typing import Sequence

import pandas as pd

from ml4ht.data.data_source import DataIndex, range_epoch_idx_generator


def df_epoch_index_generator(
    df: pd.DataFrame,
    shuffle: bool,
    training_epochs_per_true_epoch: int,
) -> Sequence[DataIndex]:
    epoch_idx_generator = range_epoch_idx_generator(
        true_epoch_len=len(df),
        shuffle=shuffle,
        training_epochs_per_true_epoch=training_epochs_per_true_epoch,
    )
    for epoch_indices in epoch_idx_generator:
        yield df.iloc[epoch_indices].reindex()
