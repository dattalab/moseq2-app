"""Groups and left-out sessions: ``config.yaml``'s ``groups`` and ``exclude``, read by every step after extraction.

A session's group is a setting, in ``config.yaml``'s ``groups``, and the
analysis uses the groups as they are now. A model keeps the groups it was fit
with, so the analysis names each session whose group has changed since the fit;
a model fit with a separate transition matrix per group keeps using its
fit-time groups for its transition analysis, and says so. Applying a model gives
each session its group from ``config.yaml`` on both front ends (1.5.0's notebook
applied without the groups, numbering the sessions instead). A session listed
under ``exclude`` is left out of every step after extraction, with one line
naming it, and a session that ``groups``, ``exclude`` or ``session_config.yaml``
names but the project does not have is a warning naming the file.
"""

from __future__ import annotations

import contextlib
import io

import h5py
import joblib
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml, write_yaml

#: Each session: its key (the recording's path in the project), its uuid, its frame count and its group.
SESSIONS = (
    ("cohort1/saline_0/depth.avi", "uuid-0", 300, "saline"),
    ("cohort1/saline_1/depth.avi", "uuid-1", 320, "saline"),
    ("cohort1/amphetamine_0/depth.avi", "uuid-2", 340, "amphetamine"),
    ("cohort1/amphetamine_1/depth.avi", "uuid-3", 360, "amphetamine"),
)
GROUPS = {key: group for key, _, _, group in SESSIONS}


def write_config(root, **top_level):
    """``config.yaml`` with a short fit in its ``model`` section and ``top_level`` (``groups``, ``exclude``)."""
    write_yaml({"camera_type": "auto", "model": {"num_iter": 2, "progressbar": False, "kappa": 1000.0},
                **top_level}, str(root / "config.yaml"))


@pytest.fixture
def project(tmp_path):
    """Four sessions of random 10-PC scores, listed in the bookkeeping file, in two groups."""
    root = tmp_path / "project"
    (root / "_pca").mkdir(parents=True)
    rng = np.random.default_rng(0)
    with h5py.File(root / "_pca" / "pca_scores.h5", "w") as f:
        for _, uuid, frames, _ in SESSIONS:
            f[f"scores/{uuid}"] = rng.normal(size=(frames, 10)).astype("float32")
            f[f"scores_idx/{uuid}"] = np.arange(frames, dtype="float32")
    write_yaml({
        "sessions": {key: {"recording": key, "folder": f"aggregate_results/s{i}", "uuid": uuid}
                     for i, (key, uuid, _, _) in enumerate(SESSIONS)},
        "pca": {"scores": "_pca/pca_scores.h5"},
    }, str(root / "moseq2-project.yaml"))
    write_config(root, groups=dict(GROUPS))
    return root


def fit(root, name="model.p", **settings):
    """Fit a model in the project through the notebook's modeling function; the model's path."""
    from moseq2_model.gui import learn_model_command

    config = read_yaml(str(root / "config.yaml"))
    config["model"].update(settings)
    write_yaml(config, str(root / "config.yaml"))
    np.random.seed(0)
    learn_model_command(str(root), name)
    return root / "models" / name


def printed(call, *args, **kwargs):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        call(*args, **kwargs)
    return stream.getvalue()


def change_group(root, key, group):
    config = read_yaml(str(root / "config.yaml"))
    config["groups"][key] = group
    write_yaml(config, str(root / "config.yaml"))


@pytest.fixture
def transition_groups(monkeypatch):
    """The group of each session the transition graph is computed with, as handed to its plotter."""
    import moseq2_viz.helpers.wrappers as wrappers

    seen = []

    def record(config_data, labels, label_group, group):
        seen.append(list(label_group))
        raise SystemExit  # stop before drawing; the groups are what the test reads

    monkeypatch.setattr(wrappers, "compute_and_graph_grouped_TMs", record)
    return seen


def plot_transitions(root, model):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project
    from moseq2_viz.options import TransitionGraphOptions
    from moseq2_viz.steps import plot_transition_graph

    def run():
        with contextlib.suppress(SystemExit):
            plot_transition_graph(session_index(Project.at(root)), str(model), str(root / "plots" / "transitions"),
                                  TransitionGraphOptions(max_syllable=5))

    return printed(run)


def test_notebook_apply_model_gives_each_session_its_group(project):
    from moseq2_model.gui import apply_model_command

    model = fit(project)
    apply_model_command(str(project), str(model), "applied.p")

    applied = joblib.load(project / "applied_models" / "applied.p")
    assert applied["metadata"]["groups"] == {uuid: group for _, uuid, _, group in SESSIONS}


def test_analysis_names_sessions_whose_group_changed_since_the_fit(project, transition_groups):
    model = fit(project)
    moved = SESSIONS[1][0]
    change_group(project, moved, "amphetamine")

    output = plot_transitions(project, model)

    assert (f"1 sessions are in a different group now than when {model} was fit: {moved} (saline then, "
            "amphetamine now). The analysis uses the groups in config.yaml.") in output
    # the transition graph groups the sessions as config.yaml does now
    assert sorted(transition_groups[0]) == ["amphetamine", "amphetamine", "amphetamine", "saline"]


def test_a_separate_transition_model_uses_its_fit_time_groups(project, transition_groups):
    model = fit(project, separate_trans=True)
    change_group(project, SESSIONS[1][0], "amphetamine")

    output = plot_transitions(project, model)

    assert (f"{model} was fit with a separate transition matrix per group, so its transition analysis uses "
            "the groups it was fit with.") in output
    assert sorted(transition_groups[0]) == ["amphetamine", "amphetamine", "saline", "saline"]


def test_an_excluded_session_is_left_out_and_named(project):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project
    from moseq2_model.gui import apply_model_command

    left_out = SESSIONS[3][0]
    write_config(project, groups=dict(GROUPS), exclude=[left_out])
    note = f"Leaving out the 1 sessions under exclude in config.yaml: {left_out}."

    output = printed(fit, project)
    model = joblib.load(project / "models" / "model.p")
    assert "uuid-3" not in model["keys"]
    assert sorted(model["keys"]) == ["uuid-0", "uuid-1", "uuid-2"]
    assert output.count(note) == 1

    output = printed(apply_model_command, str(project), str(project / "models" / "model.p"), "applied.p")
    applied = joblib.load(project / "applied_models" / "applied.p")
    assert sorted(applied["keys"]) == ["uuid-0", "uuid-1", "uuid-2"]
    assert output.count(note) == 1

    assert [f["key"] for f in session_index(Project.at(project))["files"]].count(left_out) == 0


def test_an_unknown_session_in_groups_is_a_warning(project):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project, check_session_names

    write_config(project, groups={**GROUPS, "cohort2/moved/depth.avi": "saline"}, exclude=["cohort2/gone/depth.avi"])
    write_yaml({"cohort3/elsewhere/depth.avi": {"bg_roi_index": 0}}, str(project / "session_config.yaml"))
    config, session_config = project / "config.yaml", project / "session_config.yaml"

    warnings = check_session_names(Project.at(project))

    expected = [
        f"{config} names {key}, which is not a recording in this project. It is ignored; correct or delete "
        "the entry if the recording moved."
        for key in ("cohort2/moved/depth.avi", "cohort2/gone/depth.avi")
    ] + [
        f"{session_config} names cohort3/elsewhere/depth.avi, which is not a recording in this project. It is "
        "ignored; correct or delete the entry if the recording moved."
    ]
    assert warnings == expected

    # a step reading the sessions prints them and goes on, with every session the project has
    output = printed(session_index, Project.at(project))
    for warning in expected:
        assert warning in output
    assert len(session_index(Project.at(project))["files"]) == len(SESSIONS)


def test_an_arena_entry_for_a_recording_not_yet_aggregated_is_not_a_warning(project):
    """A recording in the project that is not aggregated yet (not extracted, a SLURM job still queued) keeps
    its ``session_config.yaml`` entry without a warning: extraction uses it. An entry with no recording at its
    path is still named."""
    from moseq2_common.project import Project, check_session_names

    recording = project / "cohort2" / "queued" / "depth.avi"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"depth")
    write_yaml({"cohort2/queued/depth.avi": {"bg_roi_index": 0},
                "cohort3/elsewhere/depth.avi": {"bg_roi_index": 0}}, str(project / "session_config.yaml"))

    assert check_session_names(Project.at(project)) == [
        f"{project / 'session_config.yaml'} names cohort3/elsewhere/depth.avi, which is not a recording in this "
        "project. It is ignored; correct or delete the entry if the recording moved."
    ]


def test_add_group_without_value_matches_its_default(project):
    """``add-group`` with no ``-v`` matches its default value, ``Mouse``, instead of stopping under click 8.5."""
    from moseq2_viz.cli import cli

    for i, (key, _, _, _) in enumerate(SESSIONS):
        folder = project / "aggregate_results" / f"s{i}"
        folder.mkdir(parents=True)
        with h5py.File(folder / "results.h5", "w") as f:
            f["metadata/acquisition/SubjectName"] = "Mouse" if i == 0 else "rat"
            f["metadata/acquisition/SessionName"] = f"s{i}"

    result = CliRunner().invoke(cli, ["add-group", "-k", "SubjectName", "-g", "chosen", "-e",
                                      "--project-file", str(project / "moseq2-project.yaml")],
                                catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert read_yaml(str(project / "config.yaml"))["groups"] == {**GROUPS, SESSIONS[0][0]: "chosen"}
