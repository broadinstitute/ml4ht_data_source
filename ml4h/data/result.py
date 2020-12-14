#  Python implementation of some of rust's "Result" type
from typing import Generic, Optional, TypeVar

D = TypeVar("D")  # data type variable
E = TypeVar("E")  # error type variable


class Result(Generic[D, E]):
    def __init__(
        self,
        data: Optional[D],
        error: Optional[E],
    ):
        if data is not None and error is not None:
            raise ValueError("There cannot be data and an error in a Result.")
        if data is None and error is None:
            raise ValueError("There must be data or an error in a Result.")
        self.data = data
        self.error = error

    @classmethod
    def Data(cls, data: D):
        return cls(data, None)

    @classmethod
    def Error(cls, error: E):
        return cls(None, error)

    @property
    def ok(self) -> bool:
        return self.data is not None
