# ml4ht_data_source
`ml4ht_data_source` is a library that allows you to streamline the process
of modeling multi-modal health data.

It makes it easy to load and model on data from different storage formats
with complex QC and date time selection logic.
The data can be used in both tensorflow and pytorch.

## Library functionalities
| The library can...                                         | Example use case
------------------------------------------------------------ | -----------------
| Load data from different storage formats                   | Modeling on MRIs stored in hd5 files named by sample id with labels in a pandas data frame
| Have clear and easy to use selection of data by sample id  | Selecting a set of patients with a specific condition
| Have clear and easy to use selection of data by date-time  | Selecting ECGs that have a heart attack at most 10 days prior
| Allow flexible data transformations                        | Comparing different augmentation strategies
| Allow flexible data filtering                              | Comparing different QC strategies
| Make data exploration convenient                           | Comparing distributions of labels after different QC and date selection strategies
| Allow a random state to be shared across modalities        | Selecting a random chunk of an MRI to segment

## Setup
`ml4ht_data_source` uses python 3.6 or higher.
Setup can be done using `venv`.
```bash
python3.8 -m venv env
source env/bin/activate
pip install -r requirements.txt
pre-commit install
pip install .
```

## Tests
`ml4ht_data_source` is thoroughly tested using `pytest`.
```bash
source env/bin/activate
pip install .
pytest
```
