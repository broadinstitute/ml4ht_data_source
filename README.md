# torch_ml4h
Jamboree 2020


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
