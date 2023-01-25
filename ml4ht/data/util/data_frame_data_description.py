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
        name: str = None,
        loading_option_name: str = None,
        value_to_tensor: Callable[[Any], Tensor] = None,
    ):
        """
        DataDescription which gets data from a specific column of the provided DataFrame.

        :param df: Must be indexed with sample_id and optionally with another loading_option
        :param col: The column name to get data from
        :param name: The DataDescription name if different from the column name to get data from
        :param loading_option_name: Optional additional index column besides sample_id if needed to load data
        :param value_to_tensor: Optional Function to transform the column value into a Tensor
        """
        self._name = name or col
        self.df = df.sort_index()[col]
        self.loading_option_name = loading_option_name
        self.value_to_tensor = value_to_tensor or self._default_value_to_tensor

    @staticmethod
    def _default_value_to_tensor(x: Any) -> Tensor:
        return np.array(x)

    def get_loading_options(self, sample_id: SampleID) -> List[LoadingOption]:
        if self.loading_option_name is None:
            return [{}]
        else:
            options = self.df.loc[sample_id].index
            return [{self.loading_option_name: option} for option in options]

    @property
    def name(self) -> str:
        return self._name

    def get_raw_data(
        self,
        sample_id: SampleID,
        loading_option: LoadingOption,
    ) -> Tensor:
        if self.loading_option_name is None:
            col_val = self.df.loc[sample_id]
        else:
            col_val = self.df.loc[sample_id, loading_option[self.loading_option_name]]
        return self.value_to_tensor(col_val)
