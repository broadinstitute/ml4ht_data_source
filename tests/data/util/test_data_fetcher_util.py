import pandas as pd

from ml4ht.data.util.data_fetcher_util import df_epoch_index_generator


class TestDfEpochIndexGenerator:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["1", "2", "3"],
        },
    )

    def test_shuffle_smaller_epoch(self):
        training_steps_per_epoch = 2
        idx_generator = df_epoch_index_generator(
            self.df,
            shuffle=True,
            training_epochs_per_true_epoch=training_steps_per_epoch,
        )
        for _ in range(2):
            seen_idxs = []
            for _ in range(training_steps_per_epoch):
                seen_idxs.append(next(idx_generator))
            print(pd.concat(seen_idxs))
            pd.testing.assert_frame_equal(
                pd.concat(seen_idxs).sort_values("a"),
                self.df,
            )
