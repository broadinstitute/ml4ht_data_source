# torch_ml4h

## Jamboree 2020
We will write a data loading framework that can be used for exploring data and loading data for neural network training.

| The library should be able to                              | Example use case 
------------------------------------------------------------ | -----------------
| Load data from different storage formats                   | Modeling on MRIs stored in hd5 files named by sample id with labels in a pandas data frame
| Have clear and easy to use selection of data by sample id  | Selecting a set of patients with a specific condition
| Have clear and easy to use selection of data by date-time  | Selecting ECGs that have a heart attack at most 10 days prior
| Allow flexible data transformations                        | Cross validating different augmentation strategies
| Allow flexible data filtering                              | Cross validating different QC strategies
| Allow runtime failures in data loading                     | Applying transformations that occassionally create invalid inputs
| Make data exploration convenient                           | Comparing distributions of labels after different QC and date selection strategies
| Allow a random state to be shared across modalities        | Selecting a random chunk of an MRI to segment

## Setup
`torch_ml4h` uses python 3.8.
Setup can be done using `venv`.
```bash
python3.8 -m venv env
source env/bin/activate
pip install -r requirements.txt
pre-commit install
pip install .
```

## Tests
`torch_ml4h` uses `pytest` for tests.
```bash
pytest tests
```
