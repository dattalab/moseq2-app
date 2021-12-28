#!/usr/bin/env python
import warnings
from os.path import exists, join
from shutil import copy2

# look for moseq notebooks in the base dir
MOSEQ_NOTEBOOKS = [
    "Flip Classifier Training Notebook.ipynb",
    "MoSeq2-Extract-Modeling-Notebook.ipynb",
    "MoSeq2-Analysis-Visualization-Notebook.ipynb",
]

def check_availability(nb):
    _exists = exists(nb)
    if not _exists:
        warnings.warn(f"Notebook {nb} not found. MoSeq experience will suffer. You may need to re-download the MoSeq Docker image.")
    return _exists

MOSEQ_NOTEBOOKS = list(filter(check_availability, MOSEQ_NOTEBOOKS))

DATA_PATH = "/data"

for nb in MOSEQ_NOTEBOOKS:
    out_path = join(DATA_PATH, nb)
    if not exists(out_path):
        copy2(nb, out_path)

print(f'Data copied to: {DATA_PATH}')