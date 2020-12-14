"""Type descriptions for all data utilities"""


from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

SampleID = int
DateTime = datetime
Tensor = np.ndarray
HalfBatch = Dict[str, Tensor]  # the input or output of a batch
Batch = Tuple[HalfBatch, HalfBatch]  # a batch ready for input into an ML4H model
SampleGetter = Callable[
    [SampleID],
    Batch,
]  # a function that prepares a batch given a sample id
State = Optional[
    Dict[str, Any]
]  # a shared state across data modalities during loading a single sample id
StateSetter = Callable[
    [SampleID],
    State,
]  # a function that produces a state for a single sample id

EXCEPTIONS = (  # the exceptions caught during exploration
    IndexError,
    KeyError,
    ValueError,
    OSError,
    RuntimeError,
)
