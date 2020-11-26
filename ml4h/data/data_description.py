from abc import abstractmethod
from typing import List, Dict, Any

from ml4h.data.defines import SampleID, Tensor, DateTime


class DataDescription:
    """
    Describes a storage format of tensors.
    Each tensor should be associated with a sample id and a datetime.

    Example:
    class DictionaryDataDescription(DataDescription):
        def __init__(self):
            self.raw_data = {
                0: {
                    datetime(year=2000, month=3, day=1): 10,
                    datetime(year=2000, month=3, day=10): -1,
                },
                1: {
                    datetime(year=2020, month=12, day=25): 3,
                },
            }

        def get_dates(self, sample_id: SampleID) -> List[datetime]:
            return list(self.raw_data[sample_id])

        def get_raw_data(self, sample_id: SampleID, dt: datetime) -> Tensor:
            return self.raw_data[sample_id][dt]

        def get_summary_data(self, sample_id: SampleID, dt: datetime) -> Dict[str, Any]:
            summary = super().get_summary_data(sample_id, dt)
            summary['is_positive'] = summary['raw_data'] > 0
            return summary
    """

    @abstractmethod
    def get_dates(self, sample_id: SampleID) -> List[DateTime]:
        """Get all of the dates for one sample id"""
        pass

    @abstractmethod
    def get_raw_data(self, sample_id: SampleID, dt: DateTime) -> Tensor:
        """How to load a tensor given a sample id and a date"""
        pass

    def get_summary_data(self, sample_id: SampleID, dt: DateTime) -> Dict[str, Any]:
        """
        Get a summary of the tensor for a sample id and a date for exploration.
        It's recommended to override this for large tensors.
        """
        return {'raw_data': self.get_raw_data(sample_id, dt)}
