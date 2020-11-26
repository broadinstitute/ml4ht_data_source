from abc import abstractmethod
from typing import Dict, List, Callable
from datetime import timedelta

from ml4h.data.defines import SampleID, DateTime
from ml4h.data.data_description import DataDescription


class NoDTError(ValueError):
    """Raise when at least one DataDescription has no datetimes for a sample id"""
    pass


class DateSelector:
    """
    Selects the dates to use for input and output data for one sample id.

    Example:
    see RangeDateSelector
    """

    @abstractmethod
    def select_dates(self, sample_id: SampleID) -> Dict[DataDescription, DateTime]:
        pass

    @property
    def name(self) -> str:
        return type(self.__name__)


def first_dt(dts: List[DateTime]) -> DateTime:
    return sorted(dts)[0]


def find_closest_dt(reference_dt: DateTime, dts: List[DateTime]) -> DateTime:
    """Find the closest datetime in dts to the reference datetime"""
    return min(dts, key=lambda dt: abs(dt - reference_dt))


class RangeDateSelector(DateSelector):
    """
    Finds a set of datetimes within a range of a reference DataDescription.
    The datetime of the reference DataDescription is chosen using `reference_date_chooser`.
    The other DataDescriptions datetimes are chosen closest to the reference DataDescriptions datetime.
    """
    def __init__(
            self,
            reference_data_description: DataDescription,
            reference_date_chooser: Callable[[List[DateTime]], DateTime],
            other_data_descriptions: List[DataDescription],
            time_before: timedelta = timedelta(days=0),
            time_after: timedelta = timedelta(days=0),
    ):
        self.reference_data_description = reference_data_description
        self.reference_date_chooser = reference_date_chooser
        self.other_data_descriptions = other_data_descriptions
        self.time_before = time_before
        self.time_after = time_after

    def select_dates(self, sample_id: SampleID) -> Dict[DataDescription, DateTime]:
        ref_dts = self.reference_data_description.get_dates(sample_id)
        ref_dt = self.reference_date_chooser(ref_dts)
        min_dt = ref_dt - self.time_before
        max_dt = ref_dt + self.time_after

        all_dts = {self.reference_data_description: ref_dt}

        for data_description in self.other_data_descriptions:
            other_dts = [
                dt for dt in data_description.get_dates(sample_id)
                if min_dt <= dt <= max_dt
            ]
            if not other_dts:
                raise NoDTError('No dates found.')
            all_dts[data_description] = find_closest_dt(ref_dt, other_dts)

        return all_dts
