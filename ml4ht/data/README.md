# Data

The [data module](.) provides to main functionalities
1. [Data exploration](#exploration)
2. [Data loading for neural network training](#loading-data-for-modeling)


# Exploration
Preparing data for modeling has a few phases, all of which you can explore using [`explore`](./explore.py).
1. [Define how to access the raw data](#datadescription)
2. [Choose which dates to use](#dateselector)
2. [Apply filters and transformations to the raw data at the selected dates](#pipelinesamplegetter)

## DataDescription
[DataDescription](./data_description.py) defines how you access raw data from a storage format.
In order to use it, you implement the abstract methods of `DataDescription`.
Those are
* `get_dates`: given a sample id, what date times are available
* `get_raw_data`: given a date and sample id, what is the raw data

You can also optionally provide a name for your storage format by overriding `name`.
When you explore a `DataDescription`, you can override `get_summary_data`.
`get_summary_data` is like `get_raw_data`, except the output is a dictionary
mapping column names to what ever summary information you want from that raw data.

Once you have implemented `DataDescription`, you can explore it using `explore.explore_data_descriptions`.
Given a list of sample ids, it will give you the summary data for each sample id and date.

## DateSelector
[DateSelector](./date_selector.py) defines which dates to use from each sample id
for a set of `DataDescriptions`.
For example, you may want to select the first available data from each storage format.
In order to use `DateSelector`, you must override its abstract method
* select_dates: given a sample id, provide a date time for each `DataDescription` you want to use

You can get one epoch of selected dates and find errors in your date selector using `explore.explore_date_selector`.

## PipelineSampleGetter
Once you have working `DataDescription`s and a `DateSelector`,
you can combine them to make a full data pipeline which can be used for modeling,
called a `PipelineSampleGetter`.

`PipelineSampleGetter` is in [`sample_getter.py`](./sample_getter.py).
It allows you read raw data, select which dates to use,
and apply filters and transformations to the selected data.

Each input and output for your modeling task will be defined by a `TensorMap`.
A `TensorMap` takes a `DataDescription`
and a list of [`Transformations`](./transformation.py) to apply in order to that data.
The `Transformations` get access to the tensor data as a numpy array, the date time,
and a state shared across all `TensorMap`s for the current sample id.
A `TensorMap` can also use a summary function for exploration.

The state shared across `TensorMap`s is set by a `StateSetter`,
which is just a function that takes a sample id, and returns a dictionary
mapping strings to whatever you want to use as state.
For example, `StateSetter` can be used to randomly select the same parts of an image
for the input and outputs of a segmentation model.

Once you combine input and output `TensorMaps` with a `DateSelector`
and an optional `StateSetter`, you have a `PipelineSampleGetter`.
You can find where errors happen and get summary data using
`explore_pipeline_sample_getter`.


# Loading Data for Modeling
The point of the data loading utility is to efficiently generate data in the format ML4H models require.
ML4H models expect batches of data of format:
```python
(
    {  # input dictionary
        'input_1': batch_tensor,  # first dimension of each batch_tensor is batch size
        'input_2': batch_tensor,
        ...,
        'input_m': batch_tensor,
    },
    {  # output dictionary
        'output_1': batch_tensor,
        'output_2': batch_tensor,
        ...,
        'output_n': batch_tensor,
    }
)
```
You can see the expected python type of a batch is in [`defines.py`](./defines.py).

## Do it yourself
The most direct way to get data for a ml4ht neural network to train on is to define a `SampleGetter`.
A `SampleGetter` takes a sample id and returns an input dictionary and output dictionary.
Given a `SampleGetter`, you get a `pytorch` dataset using [`data_loader.py`](./data_loader.py).

For example, if you wanted to train an ML4H model on MNIST data:
```python
def mnist_sample_getter(sample_id):
    mnist_image = get_mnist_image_as_numpy_array(sample_id)
    mnist_label = get_mnist_label(sample_id)
    return {'image': mnist_image}, {'digit': mnist_label}
```

## Using `PipelineSampleGetter`
If you want to train your model on the data you explore, [`PipelineSampleGetter`](#pipelinesamplegetter)
is a valid `SampleGetter`, so it can be directly used in the data loading utilities.
Make sure you've explored your `PipelineSampleGetter` so that there are no sample ids
in your data that cause errors at runtime.
