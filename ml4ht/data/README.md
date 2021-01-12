# Data

The [data module](.) provides to main functionalities
1. [Data exploration](./explore.py)
2. [Data loading for neural network training](#loading-data-for-modeling)

In order to use the exploration functionality, (and other useful utilities, like datetime handling),
you implement the `DataDescription` class.

# Data loading using `DataDescription`s
Preparing data for modeling has a few phases, all of which you can explore using [`explore`](./explore.py).
1. [Define how to load data](#datadescription)
2. [Pick which loading options to use](#optionpicker)
3. [Model loading](#optionpicker)

## DataDescription
[DataDescription](./data_description.py) defines how you access raw data from a storage format.
In order to use it, you implement the abstract methods of `DataDescription`.
Those are
* `get_loading_options`: given a sample id, what loading options are available. Examples include:
    * Which dates data is available at
    * Which slices of an MRI are available
* `get_raw_data`: given a sample id and loading options, what is the raw data. Using the previous examples:
    * Given a sample id and a date, load an ECG for that date
    * Given a sample id and the slice index of, load a slice of an MRI

You can also optionally provide a name for your storage format by overriding `name`.
When you explore a `DataDescription`, you can override `get_summary_data`.
`get_summary_data` is like `get_raw_data`, except the output is a dictionary
mapping column names to what ever summary information you want from that raw data.

Once you have implemented `DataDescription`, you can explore it using `explore.explore_data_descriptions`.
Given a list of sample ids, it will give you the summary data for each sample id and loading option.

## OptionPicker
Often which loading options you pick are interdependent.
For example, if you pick slice 5 of an MRI to segment,
then you also want to pick slice 5 from the segmentation mask.

To manage interdependencies between loading options, you use an `OptionPicker`.
An option picker gets all of the available loading options for a set of `DataDescriptions`,
and decides which option to use for each.

## DataDescriptionSampleGetter
Once you have working `DataDescription`s and an `OptionPicker`,
you can combine them to make a full data pipeline which can be used for modeling,
called a `DataDescriptionSampleGetter`.
`DataDescriptionSampleGetter` is in [`sample_getter.py`](./sample_getter.py).

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

## Using `DataDescriptionSampleGetter`
If you want to train your model on the data you explore, [`DataDescriptionSampleGetter`](#pipelinesamplegetter)
is a valid `SampleGetter`, so it can be directly used in the data loading utilities.
Make sure you've explored your `DataDescriptionSampleGetter` so that there are no sample ids
in your data that cause errors at runtime.
