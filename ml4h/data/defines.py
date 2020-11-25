"""Type descriptions for all data utilities"""


from typing import Callable, Dict, Tuple, Any, Optional
from datetime import datetime

import numpy as np


SampleID = int
DateTime = datetime
Tensor = np.ndarray
TensorError = Optional[str]  # an optional failure description for a tensor processing step
HalfBatch = Dict[str, Tensor]  # the input or output of a batch
Batch = Tuple[HalfBatch, HalfBatch]  # a batch ready for input into an ML4H model
SampleGetter = Callable[[SampleID], Batch]  # a function that prepares a batch given a sample id
State = Optional[Dict[str, Any]]  # a shared state across data modalities during loading a single sample id
StateSetter = Callable[[SampleID], State]  # a function that produces a state for a single sample id

EXCEPTIONS = (  # the exceptions caught during exploration
    IndexError,
    KeyError,
    ValueError,
    OSError,
    RuntimeError,
)
