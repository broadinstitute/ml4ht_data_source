import pandas as pd

from ml4ht.data.util.data_frame_data_description import DataFrameDataDescription


TEST_DF = pd.DataFrame(
    {
        "sample_id": [0, 0, 1, 2],
        "option": ["a", "b", "a", "c"],
        "value": [1.1, 1.2, -0.2, 10.0],
    },
).set_index(["sample_id", "option"])


def test_get_loading_options():
    dd = DataFrameDataDescription(
        TEST_DF,
        "value",
    )
    for sample_id in TEST_DF.index.levels[0]:
        options = dd.get_loading_options(sample_id)
        expected_options = TEST_DF.loc[sample_id].index
        assert len(options) == len(expected_options)
        for expected_option in expected_options:
            assert any(expected_option == option["option"] for option in options)


def test_get_raw_data():
    dd = DataFrameDataDescription(
        TEST_DF,
        "value",
        lambda x: -x,
    )
    for sample_id in TEST_DF.index.levels[0]:
        options = dd.get_loading_options(sample_id)
        for option in options:
            val = dd.get_raw_data(sample_id, option)
            assert val == -TEST_DF.loc[sample_id, option["option"]][0]
