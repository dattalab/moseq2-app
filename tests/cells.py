"""Read a cell out of the notebooks a user is given.

Most cells call a library function, and their tests call the same function.
A few do their work in the cell itself, so the cell's own text is what a
test has to run or read: a transcription would go on passing after the
notebook changed under it.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"

ANALYSIS = "MoSeq2-Analysis-Visualization-Notebook.ipynb"


def code_cell(notebook, number):
    """The source of a notebook's ``number``-th code cell, counting from 1.

    Numbered the way the test modules are: markdown cells do not count.
    """
    cells = json.loads((NOTEBOOKS / notebook).read_text(encoding="utf-8"))["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    return "".join(code[number - 1]["source"])
