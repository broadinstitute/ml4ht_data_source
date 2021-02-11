from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, List

import pandas as pd

from ml4ht.data.data_description import DataDescription
from ml4ht.data.defines import EXCEPTIONS, SampleID, SampleGetter
from ml4ht.data.sample_getter import DataDescriptionSampleGetter

ERROR_COL = "error"
NO_LOADING_OPTIONS_ERROR = ValueError("No loading options")
SAMPLE_ID_COL = "sample_id"
DATA_DESCRIPTION_COL = "data_description"
LOADING_OPTION_COL = "state"


def build_df(
    summarizer: Callable[[SampleID], pd.DataFrame],
    sample_ids: List[SampleID],
    multiprocess_workers: int = 0,
) -> pd.DataFrame:
    """
    Apply a function (summarizer) to a list of sample ids to build a DataFrame
    If multiprocess_workers is 0, does not multiprocess
    """
    pool = None
    try:
        if multiprocess_workers:
            pool = Pool(multiprocess_workers)
            mapper = pool.imap_unordered(summarizer, sample_ids)
        else:
            mapper = map(summarizer, sample_ids)

        dfs = []
        for i, df in enumerate(mapper):
            dfs.append(df)
            print(f"{(i + 1) / len(sample_ids):.1%} done.", end="\r")
    finally:
        if pool is not None:
            pool.close()
    df = pd.concat(dfs)

    if ERROR_COL not in df:  # in case there are no errors, add an empty error column
        df[ERROR_COL] = None
    return df


def _format_exception(exception: Exception) -> str:
    return repr(exception)


# data description exploration
def _data_description_summarize_sample_id(
    sample_id: SampleID,
    data_description: DataDescription,
) -> pd.DataFrame:
    """
    Get the summary data for a data description for a single sample id.
    Catches and records errors.
    """
    try:
        loading_options = data_description.get_loading_options(sample_id)
        if not loading_options:
            raise NO_LOADING_OPTIONS_ERROR
    except EXCEPTIONS as e:
        return pd.DataFrame(
            {
                SAMPLE_ID_COL: [sample_id],
                DATA_DESCRIPTION_COL: [data_description.name],
                ERROR_COL: [_format_exception(e)],
            },
        )
    out = []
    for loading_options in loading_options:
        summary = loading_options
        try:
            summary = {
                **summary,
                **data_description.get_summary_data(sample_id, loading_options),
            }
        except EXCEPTIONS as e:
            summary[ERROR_COL] = _format_exception(e)
        out.append(summary)
    out = pd.DataFrame(out)
    out[DATA_DESCRIPTION_COL] = data_description.name
    out[SAMPLE_ID_COL] = sample_id
    return out


def _data_descriptions_summarize_sample_id(
    sample_id: SampleID,
    data_descriptions: List[DataDescription],
) -> pd.DataFrame:
    """
    Get the summary data for a list of DataDescriptions for a single sample id
    """
    return pd.concat(
        [
            _data_description_summarize_sample_id(sample_id, data_description)
            for data_description in data_descriptions
        ],
    )


def explore_data_descriptions(
    data_descriptions: List[DataDescription],
    sample_ids: List[SampleID],
    multiprocess_workers: int = 0,
) -> pd.DataFrame:
    """
    Get summary data of DataDescriptions for a list of sample ids
    """
    summarize = partial(
        _data_descriptions_summarize_sample_id,
        data_descriptions=data_descriptions,
    )
    return build_df(
        summarize,
        sample_ids,
        multiprocess_workers,
    )


# sample getter explore
def _pipeline_sample_getter_summarize_sample_id(
    sample_id: SampleID,
    sample_getter: SampleGetter,
) -> pd.DataFrame:
    """
    Get the summary, dates, states or errors from each TensorMap for a sample id
    """
    out = {SAMPLE_ID_COL: sample_id}
    try:
        data = sample_getter(sample_id)
        for name, tensor in {
            **data[0],
            **data[1],
        }.items():
            out[f"{name}_mean"] = [tensor.mean()]
            out[f"{name}_std"] = [tensor.std()]
            out[f"{name}_min"] = [tensor.min()]
            out[f"{name}_max"] = [tensor.max()]
            out[f"{name}_argmax"] = [tensor.argmax()]
            out[f"{name}_shape"] = [tensor.shape]
    except EXCEPTIONS as e:
        out[ERROR_COL] = [_format_exception(e)]
    out = pd.DataFrame(out)
    return pd.DataFrame(out)


def explore_sample_getter(
    sample_getter: DataDescriptionSampleGetter,
    sample_ids: List[SampleID],
    multiprocess_workers: int = 0,
) -> pd.DataFrame:
    """
    Get the datetime selected,
    """
    summarize = partial(
        _pipeline_sample_getter_summarize_sample_id,
        sample_getter=sample_getter,
    )
    return build_df(
        summarize,
        sample_ids,
        multiprocess_workers,
    )
