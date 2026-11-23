"""Head/tail detection: which classifier a project's extractions use, and its download.

``moseq2_extract.project.head_tail_detection``: the classifier
``extract.flip_classifier`` names; else the one trained in the project
(``flip_classifier_trained.pkl``); else the published one for the project's
animals and camera. The published one downloads into the project folder just
before it is first needed, once (``ensure_head_tail_detection``); a failed
download says what to do. The download is stood in for throughout.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from moseq2_common.yaml import read_yaml, write_config
from moseq2_extract import project as extraction
from moseq2_extract.steps import FLIP_CLASSIFIERS
from extract_projects import CLASSIFIER, small_project


def _set(project, **top_level):
    data = read_yaml(str(project.config_file))
    data.update(top_level)
    write_config(data, str(project.config_file))


@pytest.fixture
def downloads(monkeypatch, raw_sessions):
    """Stand in for the download: each call copies the fixture's classifier to the target and is counted."""
    calls = []

    def urlretrieve(url, target):
        calls.append(url)
        shutil.copy(Path(raw_sessions) / CLASSIFIER, target)

    monkeypatch.setattr("urllib.request.urlretrieve", urlretrieve)
    return calls


def test_downloads_just_before_first_use_and_only_once(tmp_path, raw_sessions, downloads):
    project = small_project(tmp_path / "project", raw_sessions, classifier=False, num_frames=30)
    detection = extraction.head_tail_detection(project)
    assert (detection.source, detection.label) == ("published", "adult C57 mice")
    assert detection.path == Path(project.folder) / "flip_classifier_k2_c57_10to13weeks.npz"

    extraction.plan(project)
    assert downloads == []
    extraction.extract(project)
    assert downloads == [FLIP_CLASSIFIERS["adult male c57s (K2)"]]
    assert detection.path.is_file()
    extraction.extract(project)
    assert len(downloads) == 1


def test_a_trained_classifier_in_the_project_is_used_in_place_of_the_animals_choice(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, classifier=False)
    _set(project, animals="fiber")
    trained = Path(project.folder) / "flip_classifier_trained.pkl"
    trained.write_bytes(b"a trained classifier")

    detection = extraction.head_tail_detection(project)
    assert (detection.source, detection.path, detection.label) == ("trained", trained, "your trained classifier")


def test_the_flip_classifier_setting_wins(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions)
    (Path(project.folder) / "flip_classifier_trained.pkl").write_bytes(b"a trained classifier")

    detection = extraction.head_tail_detection(project)
    assert (detection.source, detection.path) == ("setting", Path(project.folder) / CLASSIFIER)


@pytest.mark.parametrize("animals, published, label", [
    ("c57", "adult male c57s (K2)", "adult C57 mice"),
    ("other", "adult male c57s (K2)", "adult C57 mice"),
    ("fiber", "large mice with fibers (K2)", "large mice with fiber implants"),
    ("inscopix", "mice with Inscopix cables (K2)", "mice with Inscopix cables"),
])
def test_other_animals_get_adult_c57_detection(tmp_path, raw_sessions, animals, published, label):
    project = small_project(tmp_path / "project", raw_sessions, classifier=False)
    _set(project, animals=animals)

    detection = extraction.head_tail_detection(project)
    assert (detection.url, detection.label) == (FLIP_CLASSIFIERS[published], label)


@pytest.mark.parametrize("animals", ["c57", "other", "fiber", "inscopix"])
def test_azure_gets_the_azure_classifier_for_every_animals_choice(tmp_path, raw_sessions, animals):
    project = small_project(tmp_path / "project", raw_sessions, classifier=False)
    _set(project, animals=animals, camera_type="azure")

    assert extraction.head_tail_detection(project).url == FLIP_CLASSIFIERS["adult male c57s (Azure)"]


def test_a_failed_download_says_what_to_do(tmp_path, raw_sessions, monkeypatch):
    project = small_project(tmp_path / "project", raw_sessions, classifier=False)

    def urlretrieve(url, target):
        Path(target).write_bytes(b"half")
        raise OSError("Network is unreachable")

    monkeypatch.setattr("urllib.request.urlretrieve", urlretrieve)
    with pytest.raises(extraction.HeadTailDownloadError) as failed:
        extraction.extract(project)

    error = failed.value
    assert error.title == "Couldn't get head/tail detection for adult C57 mice"
    assert error.message == ("The download didn't finish (Network is unreachable). The arena tool's previews and "
                             "extraction can't run without it.")
    assert error.fix == "check this computer's internet connection, then run this cell again."
    assert error.command_line_fix == "check this computer's internet connection and run the command again."
    assert error.help_page == "Why do you need to know what my animals are?"
    assert sorted(p.name for p in Path(project.folder).iterdir() if not p.name.startswith(".")) == [
        "config.yaml", "saline_example_0"]
