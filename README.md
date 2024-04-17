# Setup

## With pipenv:
1. Make sure pipenv is installed globally: `pip install pipenv`
2. Install requirements with `pipenv install`

## With venv
1. Create virtual environment `python -m venv .venv`
2. Install requirements with `pip install -r requirements.txt`

# Dataset
Download from https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1 and extract in main project folder.


If you want to extract features, create 2 files in the root dataset folder: `all.txt` and `bad.txt`. Copy and paste the contents of `train.txt`, `test.txt`, and `val.txt` into `all.txt`. Leave `bad.txt` blank. The files names that are to be extracted are read from `all.txt`, while `bad.txt` keeps a log of any files that were skipped over.
