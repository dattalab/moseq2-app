#!/usr/bin/env python
import warnings
from os.path import exists, join
from os import makedirs
from shutil import copy2

# look for moseq notebooks in the base dir
MOSEQ_NOTEBOOKS = [
    "Flip-Classifier-Training-Notebook.ipynb",
    "MoSeq2-Extract-Modeling-Notebook.ipynb",
    "MoSeq2-Analysis-Visualization-Notebook.ipynb"
]

# look for download scripts in the base dir
# DOWNLOAD_SCRIPTS = ["download_extracted_model_results.sh",
#                     "download_small_dataset.sh",
#                     "install_moseq2_app.sh",
#                     "download_extracted_full_dataset.sh",
#                     "download_full_dataset.sh",
#                     "download_test_dataset.sh"
# ]

def check_availability(nb):
    _exists = exists(nb)
    if not _exists:
        warnings.warn(f"Notebook {nb} not found. MoSeq experience will suffer. You may need to re-download the MoSeq Docker image.")
    return _exists

MOSEQ_NOTEBOOKS = list(filter(check_availability, MOSEQ_NOTEBOOKS))
# DOWNLOAD_SCRIPTS =list(filter(check_availability, DOWNLOAD_SCRIPTS))

DATA_PATH = "/data"

for nb in MOSEQ_NOTEBOOKS:
    out_path = join(DATA_PATH, nb)
    if not exists(out_path):
        copy2(nb, out_path)

makedirs(join(DATA_PATH, 'scripts'), exist_ok=True)

# for script in DOWNLOAD_SCRIPTS:
#     out_path = join(DATA_PATH, 'scripts', script)
#     if not exists(out_path):
#         copy2(script, out_path)

print(f'Data copied to: {DATA_PATH}')
