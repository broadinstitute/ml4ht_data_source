from typing import Callable, Dict, List

from ml4ht.data.data_description import DataDescription
from ml4ht.data.defines import (
    Batch,
    HalfBatch,
    SampleID,
    LoadingOption,
)


OptionPicker = Callable[
    [
        SampleID,
        List[DataDescription],
    ],
    Dict[DataDescription, LoadingOption],
]  # a function that picks which loading options to use for DataDescriptions


class DataDescriptionSampleGetter:
    """
    DataDescriptionSampleGetter encompasses the full pipeline
    of preparing data to train a neural network for a single sample id.
    """

    def __init__(
        self,
        input_data_descriptions: List[DataDescription],
        output_data_descriptions: List[DataDescription],
        option_picker: OptionPicker = None,
    ):
        self.input_data_descriptions = input_data_descriptions
        self.output_data_descriptions = output_data_descriptions
        self.option_picker = option_picker or self._default_option_picker

    @staticmethod
    def _default_option_picker(
        sample_id: int,
        data_descriptions: List[DataDescription],
    ) -> Dict[DataDescription, LoadingOption]:
        return {
            data_description: data_description.get_loading_options(sample_id)[0]
            for data_description in data_descriptions
        }

    def _half_batch(
        self,
        sample_id: SampleID,
        loading_options: Dict[DataDescription, LoadingOption],
        is_input: bool,
    ) -> HalfBatch:
        half_batch = {}
        dds = (
            self.input_data_descriptions if is_input else self.output_data_descriptions
        )
        for data_description in dds:
            loading_option = loading_options[data_description]
            half_batch[data_description.name] = data_description.get_raw_data(
                sample_id,
                loading_option,
            )
        return half_batch

    def __call__(self, sample_id: SampleID) -> Batch:
        loading_options = self.option_picker(
            sample_id,
            self.input_data_descriptions + self.output_data_descriptions,
        )
        tensors_in = self._half_batch(sample_id, loading_options, True)
        tensors_out = self._half_batch(sample_id, loading_options, False)
        return tensors_in, tensors_out
