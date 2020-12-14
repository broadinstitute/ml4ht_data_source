from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

from ml4h.data.data_description import DataDescription
from ml4h.data.date_selector import DateSelector
from ml4h.data.defines import (
    EXCEPTIONS,
    Batch,
    DateTime,
    HalfBatch,
    SampleID,
    State,
    StateSetter,
    Tensor,
)
from ml4h.data.result import Result
from ml4h.data.transformation import Transformation


@dataclass
class TensorData:
    summary: Tensor
    dt: DateTime


ExploreTensor = Result[TensorData, str]


@dataclass
class BatchData:
    in_batch: Dict[str, ExploreTensor]
    out_batch: Dict[str, ExploreTensor]
    state: State


ExploreBatch = Result[BatchData, str]


def format_error(exception: Exception, error_description: str) -> str:
    """
    Prepare a descriptive error string for exploration.
    """
    return f"{error_description} causing error of type {type(exception).__name__}."


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
        summarizer: Callable[[Tensor], Any] = None,
    ):
        self.transformations = transformations or []
        self._data_description = data_description
        self._name = name
        # Default to taking the mean across the channel axis to summarize
        self.summarizer = summarizer or (lambda x: x.mean(axis=-1))

    @property
    def data_description(self) -> DataDescription:
        return self._data_description

    @property
    def name(self) -> str:
        return self._name

    def get_tensor_explore(
        self, sample_id: SampleID, dt: DateTime, state: State
    ) -> ExploreTensor:
        """
        For use during exploration.
        Catches errors and returns where they happen in the series of transformations.
        """
        try:
            x = self.data_description.get_raw_data(sample_id, dt)
        except EXCEPTIONS as e:
            return ExploreTensor.Error(format_error(e, f"Getting raw data failed"))
        for transformation in self.transformations:
            try:
                x = transformation(x, dt, state)
            except EXCEPTIONS as e:
                return ExploreTensor.Error(
                    format_error(e, f"{transformation.name} failed")
                )
        return ExploreTensor.Data(TensorData(self.summarizer(x), dt))

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
        return f"input_{self.name}"

    @property
    def output_name(self) -> str:
        return f"output_{self.name}"


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

    def _half_batch(
        self,
        sample_id: SampleID,
        dts: Dict[DataDescription, DateTime],
        state: State,
        is_input: bool,
        explore: bool = False,
    ) -> Union[HalfBatch, Dict[str, ExploreTensor]]:
        half_batch = {}
        tmaps = self.tensor_maps_in if is_input else self.tensor_maps_out
        for tensor_map in tmaps:
            data_description = tensor_map.data_description
            dt = dts[data_description]
            name = tensor_map.input_name if is_input else tensor_map.output_name
            if explore:
                half_batch[name] = tensor_map.get_tensor_explore(sample_id, dt, state)
            else:
                half_batch[name] = tensor_map.get_tensor(sample_id, dt, state)
        return half_batch

    def __call__(self, sample_id: SampleID) -> Batch:
        dts = self.date_selector.select_dates(sample_id)
        state = self.state_setter(sample_id)
        tensors_in = self._half_batch(sample_id, dts, state, True)
        tensors_out = self._half_batch(sample_id, dts, state, False)
        return tensors_in, tensors_out

    def explore_batch(self, sample_id: SampleID) -> ExploreBatch:
        try:
            dts = self.date_selector.select_dates(sample_id)
        except EXCEPTIONS as e:
            return ExploreBatch.Error(format_error(e, f"Selecting dates failed"))
        try:
            state = self.state_setter(sample_id)
        except EXCEPTIONS as e:
            return ExploreBatch.Error(format_error(e, f"Setting state failed"))

        tensors_in = self._half_batch(sample_id, dts, state, True, explore=True)
        tensors_out = self._half_batch(sample_id, dts, state, False, explore=True)

        return ExploreBatch.Data(BatchData(tensors_in, tensors_out, state))
