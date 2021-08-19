from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, List, TypeVar, Dict

import pandas as pd
import numpy as np

from ml4ht.data.data_description import DataDescription
from ml4ht.data.defines import EXCEPTIONS, SampleID, SampleGetter
from ml4ht.data.sample_getter import DataDescriptionSampleGetter
from ml4ht.data.data_source import DataSource, DataIndex

ERROR_COL = "error"
NO_LOADING_OPTIONS_ERROR = ValueError("No loading options")
SAMPLE_ID_COL = "sample_id"
DATA_DESCRIPTION_COL = "data_description"
LOADING_OPTION_COL = "state"


T = TypeVar("T")


def build_df(
    summarizer: Callable[[T], pd.DataFrame],
    sample_ids: List[T],
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


def _summarize_tensor(tensor: np.ndarray) -> Dict[str, float]:
    out = {}
    out["mean"] = tensor.mean()
    out["median"] = np.median(tensor)
    out["std"] = tensor.std()
    out["min"] = tensor.min()
    out["max"] = tensor.max()
    out["argmax"] = tensor.argmax()
    out["shape"] = tensor.shape
    return out


# sample getter explore
def _pipeline_sample_getter_summarize_sample_id(
    sample_id: SampleID,
    sample_getter: SampleGetter,
) -> pd.DataFrame:
    """
    Get the summary, dates, states or errors from each input and output for a sample id
    """
    out = {SAMPLE_ID_COL: sample_id}
    try:
        data = sample_getter(sample_id)
        for name, tensor in {
            **data[0],
            **data[1],
        }.items():
            tensor_summary = _summarize_tensor(tensor)
            for field, val in tensor_summary.items():
                out[f"{name}_{field}"] = [val]
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
    Summarize the values and errors of a sample getter
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


def _data_source_auto_summarize_sample_id(
    data_idx: DataIndex,
    data_source: DataSource,
) -> pd.DataFrame:
    out = {f"idx: {k}": [v] for k, v in data_idx.items()}
    try:
        data_in, data_out = data_source(data_idx)
        for name, tensor in data_in.items():
            out.update(
                {
                    f"input: {name}_{field}": [val]
                    for field, val in _summarize_tensor(tensor).items()
                },
            )
        for name, tensor in data_out.items():
            out.update(
                {
                    f"output: {name}_{field}": [val]
                    for field, val in _summarize_tensor(tensor).items()
                },
            )
    except EXCEPTIONS as e:
        out[ERROR_COL] = [_format_exception(e)]
    return pd.DataFrame(out)


def _data_source_summarize_sample_id(
    data_idx: DataIndex,
    data_source: DataSource,
) -> pd.DataFrame:
    out = {f"idx: {k}": [v] for k, v in data_idx.items()}
    try:
        data_in, data_out = data_source(data_idx)
        for name, tensor in data_in.items():
            out[f"input: {name}"] = [tensor]
        for name, tensor in data_out.items():
            out[f"output: {name}"] = [tensor]
    except EXCEPTIONS as e:
        out[ERROR_COL] = [_format_exception(e)]
    return pd.DataFrame(out)


def explore_data_source(
    data_source: DataSource,
    data_indices: List[DataIndex],
    auto_summarize=True,
    multiprocess_workers: int = 0,
) -> pd.DataFrame:
    """
    Get DataSource values over all the data indices.
    `auto_summarize` converts those values into summary statistics
    in the output `DataFrame`.
    """
    summarize = partial(
        _data_source_auto_summarize_sample_id
        if auto_summarize
        else _data_source_summarize_sample_id(),
        data_source=data_source,
    )
    return build_df(
        summarize,
        data_indices,
        multiprocess_workers,
    )
