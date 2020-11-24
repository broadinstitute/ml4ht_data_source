# TODO TESTSSS
# Data

The data module provides to main functionalities
1. [Data exploration](#exploration)
2. [Data loading for neural network training](#loading)


# Exploration

# Loading
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


The most direct way to get data for a ml4h neural network to train on is to define a `SampleGetter`.
A `SampleGetter` takes a sample id and returns an input dictionary and output dictionary.
Given a `SampleGetter`, you get a `pytorch` dataset using [`data_loader.py`](./data_loader.py).

For example, if you wanted to train an ML4H model on MNIST data:
```python
def mnist_sample_getter(sample_id):
    mnist_image = get_mnist_image_as_numpy_array(sample_id)
    mnist_label = get_mnist_label(sample_id)
    return {'image': mnist_image}, {'digit': mnist_label}

```
