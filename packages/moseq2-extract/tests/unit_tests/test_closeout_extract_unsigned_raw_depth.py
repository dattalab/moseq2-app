"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest


def test_raw_depth_cli_default_is_unsigned():
    from moseq2_extract.cli import cli
    option = next(p for p in cli.commands["extract"].params if p.name == "movie_dtype")
    assert np.dtype(option.default).kind == "u"
