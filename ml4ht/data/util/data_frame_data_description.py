from typing import Callable, Any, List

import numpy as np
import pandas as pd

from ml4ht.data.data_description import DataDescription
from ml4ht.data.defines import Tensor, SampleID, LoadingOption


class DataFrameDataDescription(DataDescription):
    def __init__(
        self,
        df: pd.DataFrame,
        col: str,
        process_col: Callable[[Any], Tensor] = None,
        name: str = None,
    ):
        """
        Gets data from a column of the provided DataFrame.
        :param df: Must be multi-indexed with sample_id, loading_option
        # TODO: allow multiple loading options
        :param col: The column name to get data from
        :param process_col: Function to turn the column value into Tensor
        """
        self.process_col = process_col or self._default_process_call
        self.df = df.sort_index()[col]
        self.option_name = self.df.index.names[1]
        self._name = name or col

    @staticmethod
    def _default_process_call(x: Any) -> Tensor:
        return np.array(x)

    def get_loading_options(self, sample_id: SampleID) -> List[LoadingOption]:
        options = self.df.loc[sample_id].index
        return [{self.option_name: option} for option in options]

    @property
    def name(self) -> str:
        return self._name

    def get_raw_data(
        self,
        sample_id: SampleID,
        loading_option: LoadingOption,
    ) -> Tensor:
        col_val = self.df.loc[sample_id, loading_option[self.option_name]]
        return self.process_col(col_val)
