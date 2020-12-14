# torch_ml4h
`torch_ml4h` is a library that allows you to streamline the process
of modeling multi-modal health data.

It makes it easy to load data and model on from different storage formats
with complex QC and date time selection logic.

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
`torch_ml4h` is thoroughly tested using `pytest`.
```bash
source env/bin/activate
pip install .
pytest
```
