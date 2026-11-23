"""What a run of extraction will do: new recordings, ones to redo and why, ones up to date.

``moseq2_extract.project.plan`` compares each included recording's extraction
record with the options the project resolves for it now. A rerun redoes only new
recordings and those whose settings, head/tail detection or recording changed;
what extraction detects for itself never counts as a change, a recording moved
away after extraction is not a change, and an extraction carrying no record is
redone, saying why. The records here are written by the test (``recorded_extraction``):
no extraction runs except where a test names a recording to extract.
"""

from __future__ import annotations

import shutil

from moseq2_common.bookkeeping import load
from moseq2_common.yaml import read_yaml, write_config, write_yaml
from moseq2_extract import project as extraction
from extract_projects import recorded_extraction, small_project

FOUR = ("amphetamine_example_0", "amphetamine_example_1", "saline_example_0", "saline_example_1")


def _keys(names):
    return [f"{name}/depth.avi" for name in names]


def _plans(project):
    return {p.key: (p.plan, p.reason) for p in extraction.plan(project)}


def test_a_first_run_plans_every_included_recording_as_new(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, FOUR)

    assert _plans(project) == {key: ("new", "") for key in _keys(FOUR)}
    assert not (tmp_path / "project" / "aggregate_results").exists()
    assert not (tmp_path / "project" / "moseq2-project.yaml").exists()


def test_a_rerun_redoes_only_new_recordings_and_changed_settings(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, FOUR)
    new, adjusted, *unchanged = _keys(FOUR)
    for key in (adjusted, *unchanged):
        recorded_extraction(project, key)
    write_yaml({adjusted: {"noise_tolerance": 45}}, str(project.session_config_file))

    assert _plans(project) == {new: ("new", ""), adjusted: ("redo", "its arena was adjusted"),
                               **{key: ("done", "") for key in unchanged}}


def test_a_setting_changed_in_the_config_is_named(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, FOUR[:2])
    one, two = _keys(FOUR[:2])
    recorded_extraction(project, one)
    recorded_extraction(project, two)
    data = read_yaml(str(project.config_file))
    data["extract"]["crop_size"] = [100, 100]
    write_config(data, str(project.config_file))
    plans = {p.key: p for p in extraction.plan(project)}
    assert (plans[one].plan, plans[one].reason) == ("redo", "crop_size changed")

    data["extract"]["max_height"] = 90
    write_config(data, str(project.config_file))
    plans = {p.key: p for p in extraction.plan(project)}
    assert (plans[two].plan, plans[two].reason, plans[two].changed) == ("redo", "2 settings changed",
                                                                        ("max_height", "crop_size"))
    assert extraction.redo_line(plans[two]) == (f"{two}: extracted again, because 2 settings changed (max_height, "
                                                "crop_size).")


def test_detected_settings_do_not_count_as_changed(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, FOUR[:2])
    depth, camera = _keys(FOUR[:2])
    # the floor's depth range found from the recording, with manual_set_depth_range off
    recorded_extraction(project, depth, options={"bg_roi_depth_range": (600.0, 700.0)})
    # the camera told from the recording, with camera_type auto
    recorded_extraction(project, camera, options={"camera_type": "kinect", "bg_roi_weights": (1, 0.1, 1),
                                                  "pixel_format": "gray16le", "movie_dtype": "<u2"})

    assert _plans(project) == {depth: ("done", ""), camera: ("done", "")}


def test_an_extraction_with_no_record_is_redone_and_says_why(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, FOUR[:1])
    (key,) = _keys(FOUR[:1])
    recorded_extraction(project, key, record=False)

    (planned,) = extraction.plan(project)
    assert (planned.plan, planned.reason) == ("redo", "no record of how it was made")
    assert extraction.redo_line(planned) == (
        f"{key}: extracted again, because nothing records how its extraction was made (an older MoSeq2, "
        "`moseq2 standalone` or a hand copy made it).")


def test_unticked_recordings_are_left_out(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, FOUR)
    data = read_yaml(str(project.config_file))
    data["exclude"] = _keys(FOUR[:2])
    write_config(data, str(project.config_file))

    assert sorted(_plans(project)) == _keys(FOUR[2:])
    assert [key for key, _ in extraction.included_recordings(project)] == _keys(FOUR[2:])


def test_a_recording_deleted_after_extraction_stays_up_to_date(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, FOUR[:2])
    kept, moved = _keys(FOUR[:2])
    recorded_extraction(project, kept)
    recorded_extraction(project, moved)
    shutil.rmtree(tmp_path / "project" / FOUR[1])

    assert _plans(project) == {kept: ("done", ""), moved: ("done", "")}
    run = extraction.extract(project)
    assert (run.extracted, run.failed) == ([], {})
    assert sorted(run.up_to_date) == [kept, moved]
    assert moved in load(project).sessions


def test_naming_a_recording_extracts_it_even_when_up_to_date(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, FOUR[:2])
    first = extraction.extract(project)
    assert sorted(first.extracted) == _keys(FOUR[:2])
    assert {p.plan for p in extraction.plan(project)} == {"done"}

    (key,) = _keys(FOUR[:1])
    again = extraction.extract(project, recordings=[key])
    assert again.extracted == [key]
    assert again.up_to_date == _keys(FOUR[1:2])
