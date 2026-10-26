"""Cell 8: watch the extracted movies.

    from moseq2_app.main import preview_extractions

    preview_extractions(progress_paths['base_dir'])

The second review cell, and the one a user actually looks at. Cell 7 says
which recordings are worth checking; this one plays them. It offers a
dropdown of every extracted recording, and choosing one renders the movie
with a playback-speed slider above it.

The movie is not served from disk. ``get_extraction`` reads the whole mp4,
base64-encodes it, and writes it into the ``src`` of an HTML ``<video>``
element, so the frames travel to the browser inside the notebook document.
Encoding costs a third on top of the file, and a full-length recording's
preview runs to tens of megabytes, which is why launching the notebook
needs ``websocket_max_message_size`` raised past its default.

The cell returns nothing. What it builds is reachable two ways: the three
objects it hands to ``display``, and the bokeh models it hands to ``show``.
"""

from __future__ import annotations

import base64
import re

import ipywidgets as widgets
import pytest
from bokeh.models import Div, Slider

from moseq2_app.main import preview_extractions
from moseq2_app.roi.controller import InteractiveExtractionViewer

from tests.extract_modeling.test_05_arena_mask import RECORDINGS
from tests.extract_modeling.test_06_extract import probe, proc


@pytest.fixture
def shown(monkeypatch):
    """Record the bokeh models handed to ``show``, and render none."""
    rendered = []
    monkeypatch.setattr(
        "moseq2_app.roi.controller.show", lambda model, *a, **k: rendered.append(model)
    )
    return rendered


def test_the_cell_displays_a_selector_a_button_and_an_output(extracted, displayed):
    """1. What a user is given, and in what order."""
    assert preview_extractions(str(extracted)) is None

    assert [type(obj).__name__ for obj in displayed.objects] == [
        "Button",
        "Dropdown",
        "Output",
    ]
    assert displayed.one(widgets.Button).description == "Clear Output"


def test_the_selector_lists_every_extracted_recording(extracted, displayed):
    """2. Every recording is reachable, and points at its own movie."""
    preview_extractions(str(extracted))
    options = dict(displayed.one(widgets.Dropdown).options)

    assert sorted(options) == RECORDINGS
    for name in RECORDINGS:
        assert options[name] == str(proc(extracted, name) / "results_00.mp4")


def test_choosing_a_recording_renders_a_player(extracted, shown):
    """3. The cell's product: a video and the control above it."""
    viewer = InteractiveExtractionViewer(data_path=str(extracted))
    viewer.get_extraction(str(proc(extracted, RECORDINGS[0]) / "results_00.mp4"))

    assert [type(model).__name__ for model in shown] == ["Slider", "Div"]


def test_the_movie_travels_inside_the_page(extracted, shown):
    """4. The frames are embedded, not linked.

    A ``src`` pointing at the file would not survive the notebook being moved
    or opened from another machine, so the movie is carried in the document.
    This is also what makes the cell's output large.
    """
    movie = proc(extracted, RECORDINGS[0]) / "results_00.mp4"
    viewer = InteractiveExtractionViewer(data_path=str(extracted))
    viewer.get_extraction(str(movie))

    div = next(model for model in shown if isinstance(model, Div))
    embedded = re.search(r'src="data:video/mp4;base64, ([^"]+)"', div.text)
    assert embedded, f"no embedded movie in the player:\n{div.text[:400]}"
    assert base64.b64decode(embedded.group(1)) == movie.read_bytes()
    assert str(movie) in div.text, "the player does not name the recording"


def test_the_player_is_sized_to_the_movie(extracted, shown):
    """5. A player sized to something else crops or letterboxes the mouse."""
    movie = proc(extracted, RECORDINGS[0]) / "results_00.mp4"
    viewer = InteractiveExtractionViewer(data_path=str(extracted))
    viewer.get_extraction(str(movie))

    div = next(model for model in shown if isinstance(model, Div))
    width, height = probe(movie, entries=("width", "height"))
    assert f'height="{height}"' in div.text
    assert f'width="{width}"' in div.text


def test_the_speed_control_covers_slow_motion_and_fast(extracted, shown):
    """6. Why a user opens this cell: watching a flip or a rear at a tenth speed."""
    viewer = InteractiveExtractionViewer(data_path=str(extracted))
    viewer.get_extraction(str(proc(extracted, RECORDINGS[0]) / "results_00.mp4"))

    slider = next(model for model in shown if isinstance(model, Slider))
    assert (slider.start, slider.end, slider.value) == (0, 4, 1)
    assert slider.title == "Playback Speed"


def test_asking_for_flipped_movies_offers_none(extracted, displayed):
    """7. The flipped movies are written by a later cell that has not run.

    ``preview_extractions`` takes a ``flipped`` argument the notebook does not
    pass. Asking for them in a project without them leaves the selector empty
    rather than falling back to the ordinary movies.
    """
    preview_extractions(str(extracted), flipped=True)
    assert dict(displayed.one(widgets.Dropdown).options) == {}
