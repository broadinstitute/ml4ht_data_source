from abc import abstractmethod
from datetime import timedelta, datetime
from typing import Callable, Dict, List, Union

from ml4ht.data.data_description import DataDescription
from ml4ht.data.defines import SampleID, LoadingOption


DateTime = Union[timedelta, datetime]
DATE_OPTION_KEY = "datetime"  # the key for dates in loading_options


class NoDTError(ValueError):
    """Raise when at least one DataDescription has no datetimes for a sample id"""

    pass


def first_dt(dts: List[DateTime]) -> DateTime:
    return sorted(dts)[0]


def find_closest_dt(reference_dt: DateTime, dts: List[DateTime]) -> DateTime:
    """Find the closest datetime in dts to the reference datetime"""
    return min(dts, key=lambda dt: abs(dt - reference_dt))


class RangeDateSelector:
    """
    Finds a set of datetimes within a range of a reference DataDescription.
    The datetime of the reference DataDescription is chosen using `reference_date_chooser`.
    The other DataDescriptions datetimes are chosen closest to the reference DataDescriptions datetime.
    """

    def __init__(
        self,
        reference_data_description: DataDescription,
        reference_date_chooser: Callable[[List[DateTime]], DateTime],
        time_before: timedelta = timedelta(days=0),
        time_after: timedelta = timedelta(days=0),
    ):
        self.reference_data_description = reference_data_description
        self.reference_date_chooser = reference_date_chooser
        self.time_before = time_before
        self.time_after = time_after

    def __call__(
        self,
        sample_id: SampleID,
        data_descriptions: List[DataDescription],
    ) -> Dict[DataDescription, Dict[str, DateTime]]:
        ref_dts = [
            option[DATE_OPTION_KEY]
            for option in self.reference_data_description.get_loading_options(sample_id)
        ]
        ref_dt = self.reference_date_chooser(ref_dts)
        min_dt = ref_dt - self.time_before
        max_dt = ref_dt + self.time_after

        all_dts = {self.reference_data_description: {DATE_OPTION_KEY: ref_dt}}

        for data_description in data_descriptions:
            other_dts = [
                option[DATE_OPTION_KEY]
                for option in data_description.get_loading_options(sample_id)
                if min_dt <= option[DATE_OPTION_KEY] <= max_dt
            ]
            if not other_dts:
                raise NoDTError("No dates found.")
            all_dts[data_description] = {
                DATE_OPTION_KEY: find_closest_dt(ref_dt, other_dts),
            }

        return all_dts
