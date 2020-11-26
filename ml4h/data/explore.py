from typing import List, Callable
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd

from ml4h.data.defines import SampleID, EXCEPTIONS
from ml4h.data.data_description import DataDescription
from ml4h.data.date_selector import DateSelector
from ml4h.data.sample_getter import PipelineSampleGetter


ERROR_COL = 'error'
SAMPLE_ID_COL = 'sample_id'
DATA_DESCRIPTION_COL = 'data_description'
DT_COL = 'dt'
STATE_COL = 'state'


def data_description_summarize_sample_id(
    sample_id: SampleID, data_description: DataDescription,
) -> pd.DataFrame:
    """
    Get the summary data for a data description for a single sample id.
    Catches and records errors.
    """
    try:
        dts = data_description.get_dates(sample_id)
    except EXCEPTIONS as e:
        return pd.DataFrame({
            SAMPLE_ID_COL: sample_id,
            DATA_DESCRIPTION_COL: data_description.name,
            ERROR_COL: type(e).__name__
        })
    out = []
    for dt in dts:
        summary = {DT_COL: dt}
        try:
            summary = {**summary, **data_description.get_summary_data(sample_id, dt)}
        except EXCEPTIONS as e:
            summary[ERROR_COL] = type(e).__name__
        out.append(summary)
    out = pd.DataFrame(out)
    out[DATA_DESCRIPTION_COL] = data_description.name
    out[SAMPLE_ID_COL] = sample_id
    return out


def data_descriptions_summarize_sample_id(
        sample_id: SampleID, data_descriptions: List[DataDescription],
) -> pd.DataFrame:
    """
    Get the summary data for a list of DataDescriptions for a single sample id
    """
    return pd.concat(
        [
            data_description_summarize_sample_id(sample_id, data_description)
            for data_description in data_descriptions
        ]
    )


def build_df(
        summarizer: Callable[[SampleID], pd.DataFrame],
        sample_ids: List[SampleID],
        multiprocess: bool,
) -> pd.DataFrame:
    pool = None
    try:
        if multiprocess:
            pool = Pool(cpu_count())
            mapper = pool.imap_unordered(summarizer, sample_ids)
        else:
            mapper = map(summarizer, sample_ids)

        dfs = []
        for i, df in enumerate(mapper):
            dfs.append(df)
            print(f'{(i + 1) // len(sample_ids):.1%} done.', end='\r')
    finally:
        if pool is not None:
            pool.close()
    return pd.concat(dfs)


def explore_data_descriptions(
        data_descriptions: List[DataDescription],
        sample_ids: List[SampleID],
        multiprocess: bool = False,
) -> pd.DataFrame:
    """
    Get summary data of DataDescriptions for a list of sample ids
    """
    summarize = partial(data_descriptions_summarize_sample_id, data_descriptions=data_descriptions)
    return build_df(
        summarize,
        sample_ids,
        multiprocess
    )


def date_selector_summarize_sample_id(
        sample_id: SampleID,
        date_selector: DateSelector,
) -> pd.DataFrame:
    """
    Get the dates selected for a single sample id
    """
    try:
        dts = date_selector.select_dates(sample_id)
    except EXCEPTIONS as e:
        return pd.DataFrame({
            SAMPLE_ID_COL: [sample_id],
            ERROR_COL: [type(e).__name__],
        })
    out = {}
    for data_description, dt in dts.items():
        out[f'{DATA_DESCRIPTION_COL}_{data_description.name}'] = [dt]
    out = pd.DataFrame(out)
    out[SAMPLE_ID_COL] = sample_id
    return out


def explore_date_selector(
        date_selector: DateSelector,
        sample_ids: List[SampleID],
        multiprocess: bool = False,
) -> pd.DataFrame:
    """
    Get the dates for each DataDescription from a DateSelector for a list of sample ids
    """
    summarize = partial(date_selector_summarize_sample_id, date_selector=date_selector)
    return build_df(
        summarize,
        sample_ids,
        multiprocess
    )


def pipeline_sample_getter_summarize_sample_id(
        sample_id: SampleID,
        sample_getter: PipelineSampleGetter,
) -> pd.DataFrame:
    """
    Get the summary, dates, states or errors from each TensorMap for a sample id
    """
    explore_batch = sample_getter.explore_batch(sample_id)
    out = {SAMPLE_ID_COL: sample_id}
    if explore_batch.ok:
        out[STATE_COL] = [explore_batch.data[2]]
        for name, tensor_result in {**explore_batch.data[0], **explore_batch.data[1]}.items():
            if tensor_result.ok:
                out[f'{name}_summary'] = [tensor_result.data.summary]
                out[f'{name}_{DT_COL}'] = [tensor_result.data.dt]
            else:
                out[f'{name}_{ERROR_COL}'] = [tensor_result.error]
    else:
        out[ERROR_COL] = [explore_batch.error]
    out = pd.DataFrame(out)
    return pd.DataFrame(out)


def explore_pipeline_sample_getter(
        pipeline_sample_getter: PipelineSampleGetter,
        sample_ids: List[SampleID],
        multiprocess: bool = False,
) -> pd.DataFrame:
    """
    Get the datetime selected,
    """
    summarize = partial(pipeline_sample_getter_summarize_sample_id, sample_getter=pipeline_sample_getter)
    return build_df(
        summarize,
        sample_ids,
        multiprocess
    )

