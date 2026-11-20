"""A step that needs a person's decision raises it; the interface asks and calls again.

Neither the file layer nor the project layer prompts. Where a step needs a
person's decision (an output that already exists, a flip classifier to pick), it
raises a ``DecisionNeeded`` (``OverwriteNeeded``, ``ChoiceNeeded``) before any
work, and ``moseq2_common.interface`` asks in the command line's words, then
calls the step once more with the answer as a keyword argument. A script
that meets the exception unanswered reads what it found and how to answer it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moseq2_common.interface import answered, ask_choice, ask_overwrite
from moseq2_common.project import ChoiceNeeded, DecisionNeeded, OverwriteNeeded

FLIPS = {
    "large mice with fibers (K2)": "https://example.org/k2_large.npz",
    "adult male c57s (K2)": "https://example.org/k2_c57.npz",
    "adult male c57s (Azure)": "https://example.org/azure.npz",
}


def answers(monkeypatch, *typed):
    """Stand in for ``input``, returning ``typed`` in turn."""
    queue = list(typed)
    monkeypatch.setattr("builtins.input", lambda *prompt: queue.pop(0))
    return queue


@pytest.mark.parametrize("typed, expected", [("y", True), ("Y", True), ("n", False), ("", False)])
def test_ask_overwrite_asks_as_before(monkeypatch, capsys, tmp_path, typed, expected):
    output = tmp_path / "_pca" / "pca.h5"
    answers(monkeypatch, typed)

    assert ask_overwrite(OverwriteNeeded(output, step="train_pca", option="overwrite_pca_train")) is expected
    assert capsys.readouterr().out == (f"The file {output} already exists.\n"
                                       "Would you like to overwrite it? [y -> yes, n -> no]\n\n")


def test_ask_choice_asks_until_a_listed_number(monkeypatch, capsys):
    left = answers(monkeypatch, "-1", "9", "1")

    assert ask_choice(ChoiceNeeded(FLIPS, what="flip classifier", step="download_flip_file")) == 1
    assert left == []
    assert capsys.readouterr().out == (
        "[0] large mice with fibers (K2) ---> https://example.org/k2_large.npz\n"
        "[1] adult male c57s (K2) ---> https://example.org/k2_c57.npz\n"
        "[2] adult male c57s (Azure) ---> https://example.org/azure.npz\n"
        "Please enter a valid number listed above\n"
        "Please enter a valid number listed above\n"
    )


def test_ask_choice_asks_again_after_text_that_is_not_a_number(monkeypatch, capsys):
    answers(monkeypatch, "x", "0")

    assert ask_choice(ChoiceNeeded(FLIPS, what="flip classifier", step="download_flip_file")) == 0
    assert capsys.readouterr().out.count("Please enter a valid number listed above") == 1


def test_answered_passes_the_answer_back_by_keyword(monkeypatch, tmp_path):
    output = tmp_path / "pca.h5"
    calls = []

    def step(folder, opts, *, replace_existing=None):
        calls.append((folder, opts, replace_existing))
        if replace_existing is None:
            raise OverwriteNeeded(output, step="train_pca", option="overwrite_pca_train")
        return "replaced" if replace_existing else "kept"

    answers(monkeypatch, "n")
    assert answered(step, "folder", "opts") == "kept"
    assert calls == [("folder", "opts", None), ("folder", "opts", False)]

    def choose(output_dir, selection=None):
        if selection is None:
            raise ChoiceNeeded(FLIPS, what="flip classifier", step="download_flip_file")
        return selection

    answers(monkeypatch, "2")
    assert answered(choose, "out", selection=None) == 2


def test_answered_calls_once_when_nothing_is_asked(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *prompt: pytest.fail("nothing should be asked"))
    assert answered(lambda a, *, b: a + b, 1, b=2) == 3


def test_answered_asks_only_once(monkeypatch, tmp_path):
    """A step that raises again after being answered stops with that decision, rather than asking forever."""
    def step(*, replace_existing=None):
        raise OverwriteNeeded(tmp_path / "pca.h5", step="train_pca", option="overwrite_pca_train")

    answers(monkeypatch, "y")
    with pytest.raises(OverwriteNeeded):
        answered(step)


def test_decisions_say_what_to_do(tmp_path):
    output = tmp_path / "_pca" / "pca.h5"
    overwrite = OverwriteNeeded(output, step="train_pca", option="overwrite_pca_train")
    assert str(overwrite) == (
        f"{output} already exists, and train_pca replaces an existing output only when told to. Call it again "
        "with replace_existing=True to replace it or replace_existing=False to keep it, or pass "
        "overwrite_pca_train=True in its options to replace it every time."
    )
    assert (overwrite.output, overwrite.step, overwrite.option) == (Path(output), "train_pca", "overwrite_pca_train")
    assert overwrite.keyword == "replace_existing"

    choice = ChoiceNeeded(FLIPS, what="flip classifier", step="download_flip_file")
    assert str(choice) == (
        "download_flip_file needs a choice of flip classifier: 0 large mice with fibers (K2), 1 adult male c57s "
        "(K2), 2 adult male c57s (Azure). Call it again with selection= the number of the one to use."
    )
    assert choice.choices == FLIPS
    assert choice.keyword == "selection"
    assert isinstance(overwrite, DecisionNeeded) and isinstance(choice, DecisionNeeded)
