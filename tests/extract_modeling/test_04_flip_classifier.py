"""Cell 4: fetch a flip classifier and record it in the config.

    from moseq2_extract.gui import download_flip_command
    selection = 1
    download_flip_command(progress_paths['base_dir'], config_filepath, selection=selection)

A flip classifier tells extraction which way round the mouse is. Four are
published, indexed 0 to 3, and the notebook's default of 1 is the adult male
C57 forest for the Kinect2.

They are ``.npz`` archives rather than pickles. The classifiers published
before 2026 were scikit-learn 0.20 pickles naming ``sklearn.ensemble.forest``,
a module renamed in 0.22, so no modern scikit-learn can read them. The same
four forests were transcribed into pickle-free archives of plain arrays and
published beside the originals under new names, which is why an older
moseq2-extract still downloads exactly what it always did.

Most of this cell's behaviour is observable without a network, by recording
what the downloader was asked to fetch: which selection maps to which URL,
that a repeat call fetches again rather than skipping, and what the menu
lists. Only whether those four URLs serve a loadable classifier needs the
network, and those tests carry the ``network`` marker and sit out of pull
request runs.
"""

from __future__ import annotations

import shutil
import urllib.request

import numpy as np
import pytest

from moseq2_extract.extract.flip_classifier import load_flip_classifier
from moseq2_extract.gui import download_flip_command
from moseq2_extract.util import read_yaml

BASE = "https://moseq-data.s3.amazonaws.com/flip-classifiers/"

#: Selection index -> (label as the menu prints it, served filename).
FLIP_FILES = {
    0: ("large mice with fibers (K2)", "flip_classifier_k2_largemicewithfiber.npz"),
    1: ("adult male c57s (K2)", "flip_classifier_k2_c57_10to13weeks.npz"),
    2: ("mice with Inscopix cables (K2)", "flip_classifier_k2_inscopix.npz"),
    3: ("adult male c57s (Azure)", "flip_classifier_azure_temp.npz"),
}


@pytest.fixture
def recorded_download(monkeypatch, raw_sessions):
    """Record what would have been fetched, and put a real classifier there.

    Replaces the one network call so the rest of the cell runs for real. The
    file it writes is the classifier already in the fixture bundle, so
    anything downstream that opens the result gets something valid.
    """
    calls = []
    source = raw_sessions / "flip_classifier_k2_c57_10to13weeks.npz"

    def fake_urlretrieve(url, filename, *args, **kwargs):
        calls.append((url, filename))
        shutil.copy(source, filename)
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    return calls


# --- offline ---------------------------------------------------------------


@pytest.mark.parametrize("selection", sorted(FLIP_FILES))
def test_selection_maps_to_the_published_classifier(selection, tmp_path, recorded_download):
    """Each index fetches the classifier it is documented to fetch."""
    _, filename = FLIP_FILES[selection]
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")

    download_flip_command(str(tmp_path), str(config), selection=selection)

    assert len(recorded_download) == 1
    url, destination = recorded_download[0]
    assert url == BASE + filename
    assert destination == str(tmp_path / filename)


def test_a_negative_selection_is_rejected(tmp_path, recorded_download):
    """There is no classifier at a negative index, so `-1` fetches nothing."""
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")

    with pytest.raises(IndexError):
        download_flip_command(str(tmp_path), str(config), selection=-1)
    assert recorded_download == []


def test_a_selection_past_the_end_raises(tmp_path, recorded_download):
    """An index with no classifier behind it fails before anything is fetched."""
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")

    with pytest.raises(IndexError):
        download_flip_command(str(tmp_path), str(config), selection=len(FLIP_FILES))
    assert recorded_download == []


def test_the_prompt_asks_again_for_a_number_it_does_not_list(
    tmp_path, recorded_download, monkeypatch, capsys
):
    """With no selection the menu is printed and a number is asked for.

    A number with no classifier behind it, negative or past the end, is asked
    for again, the same as text that is not a number.
    """
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")
    answers = iter(["-1", str(len(FLIP_FILES)), "1"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    download_flip_command(str(tmp_path), str(config), selection=None)

    assert recorded_download == [(BASE + FLIP_FILES[1][1], str(tmp_path / FLIP_FILES[1][1]))]
    assert capsys.readouterr().out.count("Please enter a valid number listed above") == 2


def test_the_config_records_the_downloaded_classifier(tmp_path, recorded_download):
    """The cell's output: the config now names a classifier on disk."""
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")

    download_flip_command(str(tmp_path), str(config), selection=1)

    recorded = read_yaml(str(config))["flip_classifier"]
    assert recorded == str(tmp_path / FLIP_FILES[1][1])
    assert (tmp_path / FLIP_FILES[1][1]).exists()


def test_running_the_cell_twice_fetches_again(tmp_path, recorded_download):
    """Captured, not corrected: there is no existence check.

    A user who re-runs this cell pays for the download a second time. The
    file that results is the same one, so nothing downstream notices.
    """
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")

    download_flip_command(str(tmp_path), str(config), selection=1)
    first = (tmp_path / FLIP_FILES[1][1]).read_bytes()
    download_flip_command(str(tmp_path), str(config), selection=1)
    second = (tmp_path / FLIP_FILES[1][1]).read_bytes()

    assert len(recorded_download) == 2, "the second call did not fetch again"
    assert first == second


def test_no_selection_lists_what_is_available(tmp_path, monkeypatch, recorded_download, capsys):
    """With no selection the cell prints the menu and asks for one."""
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")
    monkeypatch.setattr("builtins.input", lambda *_: "1")

    download_flip_command(str(tmp_path), str(config), selection=None)

    out = capsys.readouterr().out
    for index, (label, filename) in FLIP_FILES.items():
        assert f"[{index}] {label} ---> {BASE + filename}" in out


def test_the_bundled_classifier_is_the_one_the_chain_uses(configured, flipped):
    """The chain reaches cell 4's state: the config names a classifier that loads."""
    assert read_yaml(str(configured))["flip_classifier"] == str(flipped)
    assert flipped.suffix == ".npz"
    load_flip_classifier(str(flipped))


# --- network ---------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("selection", sorted(FLIP_FILES))
def test_every_published_classifier_downloads_and_classifies(selection, tmp_path):
    """All four URLs serve an archive that loads and produces probabilities.

    Downloading proves the bytes arrive. Loading and classifying is what
    catches a transcription that arrived intact and is wrong inside, and
    nothing else in either notebook touches three of these four.
    """
    _, filename = FLIP_FILES[selection]
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")

    download_flip_command(str(tmp_path), str(config), selection=selection)

    downloaded = tmp_path / filename
    assert downloaded.exists() and downloaded.stat().st_size > 0
    assert read_yaml(str(config))["flip_classifier"] == str(downloaded)

    classifier = load_flip_classifier(str(downloaded))
    assert classifier.n_classes_ == 2
    assert classifier.n_features_ > 0

    frames = np.random.default_rng(0).random((4, classifier.n_features_))
    proba = classifier.predict_proba(frames)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.network
def test_refetching_over_the_network_gives_the_same_file(tmp_path):
    """Re-running the cell for real overwrites with identical content."""
    config = tmp_path / "config.yaml"
    config.write_text("flip_classifier: null\n")
    target = tmp_path / FLIP_FILES[1][1]

    download_flip_command(str(tmp_path), str(config), selection=1)
    first = target.read_bytes()
    download_flip_command(str(tmp_path), str(config), selection=1)

    assert target.read_bytes() == first
