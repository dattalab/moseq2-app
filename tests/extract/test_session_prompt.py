"""Choosing recordings at the extraction cell's prompt.

With ``extract_all = False`` the extraction cell lists the recordings it
found, numbered from 1, and asks which to extract. An answer is a list of
items separated by commas. Each item is a recording number ``N``, a range
``N-M`` including both ends, or either of those prefixed with ``e`` to leave
those recordings out. Without any item that picks recordings, the answer
starts from all of them. A recording is extracted once, however many items
name it.

An item the prompt cannot use is named in a message, and the prompt asks
again: a number that is not on the list, a range that runs backwards, an
item that is not a number or a range, or an answer that leaves nothing to
extract. ``q`` quits.
"""

from __future__ import annotations

import pytest

from moseq2_extract.gui import get_selected_sessions

RECORDINGS = ["r1", "r2", "r3", "r4"]
AGAIN = "Try again or press q to quit."


def ask(monkeypatch, capsys, answer):
    """Give the prompt ``answer``, then ``q`` if it asks again.

    Returns the recordings chosen and what the prompt printed after the list.
    """
    answers = iter([answer, "q"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    chosen = get_selected_sessions(RECORDINGS, False)
    return chosen, capsys.readouterr().out.split("Press q to quit.\n", 1)[1]


@pytest.mark.parametrize(
    "answer, numbers",
    [
        ("1,3", [1, 3]),
        ("1, 3", [1, 3]),
        ("2,1", [2, 1]),
        ("4", [4]),
        ("1-2", [1, 2]),
        ("1-4", [1, 2, 3, 4]),
        ("1,4", [1, 4]),
        ("e2", [1, 3, 4]),
        ("E2", [1, 3, 4]),
        ("e1-2", [3, 4]),
        ("e1,e2", [3, 4]),
        ("1-3,e2", [1, 3]),
        ("1,1", [1]),
        ("1-3,2", [1, 2, 3]),
    ],
)
def test_an_answer_selects_the_recordings_it_names(monkeypatch, capsys, answer, numbers):
    chosen, printed = ask(monkeypatch, capsys, answer)
    assert chosen == [RECORDINGS[n - 1] for n in numbers]
    assert printed == ""


@pytest.mark.parametrize(
    "answer, item, reason",
    [
        ("0", "0", "there is no session 0"),
        ("0-1", "0-1", "there is no session 0"),
        ("1,0", "0", "there is no session 0"),
        ("5", "5", "there is no session 5"),
        ("1,9", "9", "there is no session 9"),
        ("1-9", "1-9", "there is no session 9"),
        ("1-9,2", "1-9", "there is no session 9"),
        ("e5", "e5", "there is no session 5"),
        ("4-2", "4-2", "a range runs from the lower number to the higher"),
        ("first", "first", "not a number or a range"),
        ("1,x", "x", "not a number or a range"),
        ("1 2", "1 2", "not a number or a range"),
        ("-1", "-1", "not a number or a range"),
        ("Q2", "Q2", "not a number or a range"),
        ("e1-4", "e1-4", "that leaves no sessions to extract"),
    ],
)
def test_an_item_the_prompt_cannot_use_is_named_and_asked_again(monkeypatch, capsys, answer, item, reason):
    chosen, printed = ask(monkeypatch, capsys, answer)
    assert chosen == []
    assert printed == f'Invalid input "{item}": {reason}. {AGAIN}\n'


@pytest.mark.parametrize("answer", ["", " ", ","])
def test_an_empty_answer_asks_again(monkeypatch, capsys, answer):
    chosen, printed = ask(monkeypatch, capsys, answer)
    assert chosen == []
    assert printed == f"Invalid input. {AGAIN}\n"


@pytest.mark.parametrize("answer", ["q", "Q", " q "])
def test_q_quits(monkeypatch, capsys, answer):
    chosen, printed = ask(monkeypatch, capsys, answer)
    assert chosen == []
    assert printed == ""
