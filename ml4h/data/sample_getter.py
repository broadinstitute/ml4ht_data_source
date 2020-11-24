from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from enum import Enum

from ml4h.data.data_description import DataDescription
from ml4h.data.date_selector import DateSelector
from ml4h.data.defines import StateSetter, SampleID, DateTime, State, Tensor, TensorError, EXCEPTIONS, Batch


class TransformationType(Enum):
    FILTER = 'filter'
    AUGMENTATION = 'augmentation'
    NORMALIZATION = 'normalization'


@dataclass
class Transformation:
    """
    Transformations are applied to Tensors.
    They can filter, augment, or normalize data.
    Filter transformations should raise errors on bad data.

    Example:
    import numpy as np

    def error_on_negative(x: Tensor, _, __) -> Tensor:
        if np.any(x <= 0):
            raise ValueError('Not all values positive.')
        return x

    filter_all_positive = Transformation(TransformationType.FILTER, error_on_negative)
    """
    transformation_type: TransformationType
    transformation: Callable[[Tensor, DateTime, State], Tensor]

    @property
    def is_augmentation(self) -> bool:
        return self.transformation_type == TransformationType.AUGMENTATION

    @property
    def is_filter(self) -> bool:
        return self.transformation_type == TransformationType.FILTER

    @property
    def is_normalization(self) -> bool:
        return self.transformation_type == TransformationType.NORMALIZATION


def format_error(exception: Exception, error_description: str) -> str:
    """
    Prepare a descriptive error string for exploration.
    """
    return f'{error_description} causing error of type {type(exception).__name__}.'


class TensorMap:
    """
    Gets raw data using a data description then applies a series of transformations to the raw data.
    Can keep track of errors and filtering during exploration time.
    """
    def __init__(
            self,
            name: str,
            data_description: DataDescription,
            transformations: List[Transformation] = None,
    ):
        self.transformations = transformations or []
        self._data_description = data_description
        self._name = name

    @property
    def data_description(self) -> DataDescription:
        return self._data_description

    @property
    def name(self) -> str:
        return self._name

    def get_tensor_explore(self, sample_id: SampleID, dt: DateTime, state: State) -> Tuple[Optional[Tensor], TensorError]:
        """
        For use during exploration.
        Catches errors and returns where they happen in the series of transformations.
        """
        try:
            x = self.data_description.get_raw_data(sample_id, dt)
        except EXCEPTIONS as e:
            return None, format_error(e, f'Getting raw data failed')
        for transformation in self.transformations:
            try:
                x = transformation(x, dt, state)
            except EXCEPTIONS as e:
                return None, format_error(e, f'{transformation.__name__} failed')
        return x, None

    def get_tensor(self, sample_id: SampleID, dt: DateTime, state: State) -> Tensor:
        """
        For use during modeling. Does not catch errors.
        """
        x = self.data_description.get_raw_data(sample_id, dt)
        for transformation in self.transformations:
            x = transformation(x, dt, state)
        return x

    @property
    def input_name(self) -> str:
        return f'input_{self.name}'

    @property
    def output_name(self) -> str:
        return f'output_{self.name}'


class PipelineSampleGetter:
    """
    PipelineSampleGetter encompasses the full pipeline of preparing data to train a neural network for a single sample id.

    For a single sample id, the process is:
    1. Select a date for each DataDescription using a DateSelector
    2. Optionally define a random state shared across TensorMaps
    3. Get a dictionary of input tensors from input TensorMaps using the sample id, datetime, state
    4. Get a dictionary of output tensors from input TensorMaps using the sample id, datetime, state
    """
    def __init__(
            self,
            tensor_maps_in: List[TensorMap],
            tensor_maps_out: List[TensorMap],
            date_selector: DateSelector,
            state_setter: StateSetter = None,
    ):
        self.tensor_maps_in = tensor_maps_in
        self.tensor_maps_out = tensor_maps_out
        self.date_selector = date_selector
        self.state_setter = state_setter or (lambda sample_id: {})

    def __call__(self, sample_id: SampleID) -> Batch:
        dts = self.date_selector.select_dates(sample_id)
        state = self.state_setter(sample_id)
        tensors_in = {}
        for tensor_map in self.tensor_maps_in:
            data_description = tensor_map.data_description
            dt = dts[data_description]
            tensors_in[tensor_map.input_name] = tensor_map.get_tensor(sample_id, dt, state)

        tensors_out = {}
        for tensor_map in self.tensor_maps_out:
            data_description = tensor_map.data_description
            dt = dts[data_description]
            tensors_out[tensor_map.output_name] = tensor_map.get_tensor(sample_id, dt, state)

        return tensors_in, tensors_out

    def explore_batch(self, sample_id: SampleID) -> Tuple[Optional[Batch], TensorError]:
        try:
            dts = self.date_selector.select_dates(sample_id)
        except EXCEPTIONS as e:
            return None, format_error(e, f'Selecting dates failed')
        try:
            state = self.state_setter(sample_id)
        except EXCEPTIONS as e:
            return None, format_error(e, f'Setting state failed')

        tensors_in = {}
        for tensor_map in self.tensor_maps_in:
            data_description = tensor_map.data_description
            dt = dts[data_description]
            tensor, error = tensor_map.get_tensor_explore(sample_id, dt, state)
            if error is not None:
                return None, error
            tensors_in[tensor_map.input_name] = tensor

        tensors_out = {}
        for tensor_map in self.tensor_maps_in:
            data_description = tensor_map.data_description
            dt = dts[data_description]
            tensor, error = tensor_map.get_tensor_explore(sample_id, dt, state)
            if error is not None:
                return None, error
            tensors_in[tensor_map.output_name] = tensor

        return (tensors_in, tensors_out), None
