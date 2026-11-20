"""What the command line and the notebooks use to ask a person a step's question.

A step that needs a decision raises a ``moseq2_common.project.DecisionNeeded``
before doing any work. ``answered`` calls the step, and on that exception asks
the person in the command line's words (``ask_overwrite``, ``ask_choice``), then
calls the step once more with the answer passed as the exception's ``keyword``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import click

from moseq2_common.project import ChoiceNeeded, DecisionNeeded, OverwriteNeeded

__all__ = ["answered", "ask_choice", "ask_overwrite"]

T = TypeVar("T")


def ask_overwrite(decision: OverwriteNeeded) -> bool:
    """Ask whether to replace ``decision.output``; True when the person types ``y``."""
    click.echo(f'The file {decision.output} already exists.\nWould you like to overwrite it? [y -> yes, n -> no]\n')
    return input().lower() == 'y'


def ask_choice(decision: ChoiceNeeded) -> int:
    """List ``decision.choices`` numbered from 0 and ask until the person types one of the numbers; that number."""
    for index, (label, value) in enumerate(decision.choices.items()):
        print(f"[{index}] {label} ---> {value}")
    while True:
        try:
            choice = int(input("Enter a selection "))
        except ValueError:
            choice = -1
        if 0 <= choice < len(decision.choices):
            return choice
        print("Please enter a valid number listed above")


def _ask(decision: DecisionNeeded):
    if isinstance(decision, OverwriteNeeded):
        return ask_overwrite(decision)
    if isinstance(decision, ChoiceNeeded):
        return ask_choice(decision)
    raise decision


def answered(function: Callable[..., T], *args, **kwargs) -> T:
    """``function(*args, **kwargs)``; when it raises a ``DecisionNeeded``, the person is asked and
    ``function`` is called once more with the answer as the decision's keyword argument.

    Callers pass that keyword (``replace_existing``, ``selection``) by name, never positionally, so the
    answer replaces it.
    """
    try:
        return function(*args, **kwargs)
    except DecisionNeeded as decision:
        answer = _ask(decision)
        return function(*args, **{**kwargs, decision.keyword: answer})
