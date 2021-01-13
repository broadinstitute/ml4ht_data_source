from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ml4ht.data.defines import SampleID, Tensor, LoadingOption


class DataDescription:
    """
    Describes how to load data from a storage format of tensor data.
    Each tensor should be associated with a sample id and specific loading options.
    For an example, see tests/data/test_data_description.py
    """

    def get_loading_options(self, sample_id: SampleID) -> List[LoadingOption]:
        """
        Get all of the loading options for one sample id
        Loading options might be
        * the available date-times for a sample id
        * which slice of an MRI to use
        * a random seed to use for warping an image and its segmentation

        In the case where there are a huge number of available options,
        for example picking a random seed,
        you can just return a representative sample of them for use in exploration.
        """
        return [{}]

    @abstractmethod
    def get_raw_data(
        self,
        sample_id: SampleID,
        loading_option: LoadingOption,
    ) -> Tensor:
        """How to load a tensor given a sample id and a loading option"""
        pass

    def get_summary_data(
        self,
        sample_id: SampleID,
        loading_option: LoadingOption,
    ) -> Dict[str, Any]:
        """
        Get a summary of the tensor for a sample id and a date for exploration.
        It's recommended to override this for large tensors.
        """
        return {"raw_data": self.get_raw_data(sample_id, loading_option)}

    @property
    def name(self) -> str:
        return type(self).__name__
