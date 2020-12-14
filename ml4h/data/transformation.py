from enum import Enum
from dataclasses import dataclass
from typing import Callable

from ml4h.data.defines import DateTime, State, Tensor


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

    @property
    def name(self) -> str:
        return f'{self.transformation.__name__}_{self.transformation_type}'

    def __call__(self, tensor: Tensor, dt: DateTime, state: State) -> Tensor:
        return self.transformation(tensor, dt, state)
