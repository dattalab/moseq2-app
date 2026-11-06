"""A session's frame rate comes from its recording, never from a setting.

Extraction works out the rate from the recording's timestamps (the median gap
between them, to the nearest whole frame per second), or from the rate the
video itself declares when the timestamps give none, and writes it to the
extraction ``.h5`` as ``/metadata/fps``. Every later step reads it from there
with ``moseq2_common.timestamps.frame_rate``, as it reads the timestamps; an
extraction older than ``/metadata/fps`` gives its rate from its timestamps.

So a 15 fps recording is a 15 fps session all the way through: its preview
movie plays at 15 fps, PCA places its dropped frames and times its
changepoints at 15 fps, and ``get-best-model`` turns model syllables into
seconds at 15 fps. ``fps`` is not a setting: an ``fps:`` line in
``config.yaml`` is named in the message listing the settings no step reads,
and ignored.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner
from dask.distributed import Client

from moseq2_common.records import read_record
from moseq2_common.yaml import write_yaml

RATE = 15
N = 300


# ---------------------------------------------------------------- reading it


def test_frame_rate_reads_metadata_fps(tmp_path):
    from moseq2_common.timestamps import frame_rate

    with h5py.File(tmp_path / "s.h5", "w") as f:
        f["metadata/fps"] = np.float32(RATE)
        f["timestamps"] = np.arange(10) * 1000 / 30
        assert frame_rate(f) == RATE


def test_frame_rate_of_an_extraction_without_metadata_fps_comes_from_its_timestamps(tmp_path):
    from moseq2_common.timestamps import frame_rate

    with h5py.File(tmp_path / "s.h5", "w") as f:
        f["timestamps"] = np.delete(np.arange(N) * 1000 / RATE, 20)
        assert frame_rate(f) == RATE


def test_frame_rate_without_metadata_fps_or_timestamps_names_the_file(tmp_path):
    from moseq2_common.timestamps import frame_rate

    with h5py.File(tmp_path / "s.h5", "w") as f:
        f["frames"] = np.zeros((3, 2, 2))
        with pytest.raises(ValueError) as error:
            frame_rate(f)
    message = str(error.value)
    for named in (str(tmp_path / "s.h5"), "/metadata/fps", "timestamps", "extract the session again"):
        assert named in message, message


# ---------------------------------------------------------------- extraction writes it


@pytest.fixture
def recording_at_15_fps(tmp_path, raw_sessions):
    """A copy of one fixture recording whose depth_ts.txt puts its frames 1/15 s apart."""
    source = Path(raw_sessions) / "saline_example_0"
    session = tmp_path / "saline_example_0"
    session.mkdir()
    for name in ("depth.avi", "metadata.json"):
        shutil.copy(source / name, session / name)
    n = len((source / "depth_ts.txt").read_text().split("\n")) - 1
    start = 25044816.0
    (session / "depth_ts.txt").write_text("".join(f"{start + i * 1000 / RATE:.4f}\n" for i in range(n)))
    return session


def test_extraction_records_the_recordings_own_frame_rate(recording_at_15_fps):
    import moseq2_extract.cli as extract_cli
    from moseq2_extract.io.video import get_video_info

    output_dir = recording_at_15_fps / "proc"
    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(recording_at_15_fps / "depth.avi"), "--output-dir", str(output_dir), "--num-frames", "20"],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)

    with h5py.File(output_dir / "results.h5", "r") as f:
        assert f["metadata/fps"][()] == RATE
        assert "metadata/extraction/parameters" not in f
    assert "fps" not in read_record(output_dir / "results.h5")["options"]
    assert get_video_info(str(output_dir / "results.mp4"))["fps"] == RATE


# ---------------------------------------------------------------- PCA reads it


@pytest.fixture
def client():
    with Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as c:
        yield c


@pytest.fixture
def session_at_15_fps(tmp_path):
    """A synthetic 15 fps extraction whose camera dropped frame 20."""
    timestamps = np.delete(np.arange(N) / RATE, 20)
    h5 = tmp_path / "s.h5"
    with h5py.File(h5, "w") as f:
        f["frames"] = np.random.default_rng(0).random((len(timestamps), 4, 4))
        f["frames_mask"] = np.ones((len(timestamps), 4, 4), dtype=bool)
        f["timestamps"] = timestamps * 1000
        f["timestamps"].attrs["unit"] = "milliseconds"
        f["metadata/fps"] = np.float32(RATE)
        f["metadata/uuid"] = "s"
    return str(h5), str(h5)


def test_pca_places_dropped_frames_at_the_recordings_own_rate(tmp_path, session_at_15_fps, client):
    """The dropped frame gets its empty row, and changepoints are timed at 15 fps, not 30."""
    from moseq2_pca.pca.util import apply_pca_dask, get_changepoints_dask

    h5, yml = session_at_15_fps
    scores_file = str(tmp_path / "pca_scores")
    apply_pca_dask(
        np.ones((2, 16)), [h5], [yml], use_fft=False,
        clean_params={"gaussfilter_time": 0, "medfilter_time": [0]},
        save_file=scores_file, chunk_size=64, mask_params={}, missing_data=False,
        client=client, h5_path="/frames",
    )
    with h5py.File(scores_file + ".h5", "r") as f:
        idx = f["scores_idx/s"][()]
    assert len(idx) == N
    np.testing.assert_array_equal(np.flatnonzero(np.isnan(idx)), [20])

    cps_file = str(tmp_path / "changepoints")
    get_changepoints_dask(
        {"k": 6, "sigma": 3.5, "peak_height": 0.5, "peak_neighbors": 1, "rps": 50},
        np.random.default_rng(1).random((2, 16)), [h5], [yml],
        save_file=cps_file, chunk_size=64, mask_params={}, missing_data=False, client=client,
    )
    with h5py.File(cps_file + ".h5", "r") as f:
        assert f["metadata/fps"][()] == RATE
        assert f["cps/s"].attrs["fps"] == RATE
        times = f["cps/s"][()]
    assert len(times) > 5
    # each changepoint falls on a frame, a whole number of 1/15 s from the start
    np.testing.assert_allclose(times * RATE, np.round(times * RATE), atol=1e-4)


# ---------------------------------------------------------------- get-best-model reads it


def test_best_model_times_syllables_at_each_sessions_rate(tmp_path, monkeypatch):
    """Model labels become seconds at the rate the changepoints file records for their session."""
    import moseq2_viz.helpers.wrappers as wrappers

    cp_file = tmp_path / "changepoints.h5"
    with h5py.File(cp_file, "w") as f:
        f["metadata/fps"] = np.float32(RATE)
        f["cps/a"] = np.array([0.4, 0.4], dtype="float32")
        f["cps/a"].attrs["fps"] = np.float32(RATE)
    model = tmp_path / "models" / "m.p"
    model.parent.mkdir()
    model.write_bytes(b"")
    # one syllable 6 frames long between two others: 6 / 15 s = 0.4 s
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
    monkeypatch.setattr(wrappers, "parse_model_results",
                        lambda path: {"labels": [labels], "keys": ["a"], "model_parameters": {"kappa": 1}})
    seen = {}

    def best_fit(cp_path, model_results):
        seen.update(model_results)
        raise RuntimeError("stop after loading the models")

    monkeypatch.setattr(wrappers, "get_best_fit", best_fit)
    with pytest.raises(RuntimeError, match="stop after loading"):
        wrappers.get_best_fit_model_wrapper(str(model.parent), str(cp_file), None)
    np.testing.assert_allclose(seen[str(model)]["changepoints"], [6 / RATE])


# ---------------------------------------------------------------- fps is not a setting


def test_an_fps_key_in_config_is_named_and_ignored(tmp_path, monkeypatch):
    """An `fps:` line in the pca section is listed with the settings no step reads; apply-pca runs."""
    import moseq2_pca.steps as steps
    from moseq2_pca.cli import cli

    calls = []

    def record(input_dir, output_dir, output_file, opts, **kwargs):
        calls.append(opts)
        return steps.ApplyPCAResult(**{f.name: None for f in dataclasses.fields(steps.ApplyPCAResult)})

    monkeypatch.setattr(steps, "apply_pca", record)
    config = tmp_path / "config.yaml"
    write_yaml({"camera_type": "auto", "pca": {"fps": 20, "chunk_size": 250}}, str(config))

    result = CliRunner().invoke(cli, ["apply-pca", "-i", str(tmp_path / "aggregate_results"),
                                      "-o", str(tmp_path / "_pca"), "--config-file", str(config)])

    assert result.exit_code == 0, result.output
    (opts,) = calls
    assert opts.chunk_size == 250
    assert not hasattr(opts, "fps")
    assert "no pca step reads, so they are ignored: `fps`" in result.output, result.output


def test_extraction_settings_have_no_fps():
    from moseq2_extract.options import ExtractOptions
    from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions
    from moseq2_viz.options import BestModelOptions

    for cls in (ExtractOptions, ApplyPCAOptions, ChangepointsOptions, BestModelOptions):
        assert "fps" not in {f.name for f in dataclasses.fields(cls)}, cls.__name__


# ---------------------------------------------------------------- behavioural statistics


def test_syllable_durations_use_each_sessions_own_rate(tmp_path):
    """A syllable held for 30 frames lasts 2 s in a 15 fps session and 1 s in a 30 fps one; the statistics
    table (``moseq_df``'s per-syllable statistics, the stats plots) reads each session's rate from its
    extraction. 1.5.0 divided every session's frame counts by 30."""
    import pandas as pd

    from moseq2_viz.model.util import compute_behavioral_statistics

    rows = []
    for uuid, rate in (("slow", RATE), ("fast", 30)):
        h5 = tmp_path / f"{uuid}.h5"
        with h5py.File(h5, "w") as f:
            f["metadata/fps"] = float(rate)
        labels = [0] * 30 + [1] * 10
        onsets = [True] + [False] * 29 + [True] + [False] * 9
        for label, onset in zip(labels, onsets):
            rows.append({"group": "g", "uuid": uuid, "h5_path": str(h5), "labels (original)": label,
                         "labels (usage sort)": label, "onset": onset, "velocity_2d_mm": np.float32(1.0)})
    stats = compute_behavioral_statistics(pd.DataFrame(rows), count="usage")
    duration = stats.set_index(["uuid", "syllable"])["duration"]
    assert duration[("slow", 0)] == pytest.approx(30 / RATE)
    assert duration[("fast", 0)] == pytest.approx(1.0)
    assert duration[("slow", 1)] == pytest.approx(10 / RATE)


def test_the_flip_preview_movie_plays_at_the_sessions_rate(tmp_path, monkeypatch):
    """Applying the flip classifier writes each session's flipped preview at the session's own rate."""
    import moseq2_app.flip.widget as widget

    h5 = tmp_path / "s" / "results.h5"
    h5.parent.mkdir()
    with h5py.File(h5, "w") as f:
        f["frames"] = np.zeros((4, 8, 8), dtype="uint8")
        f["scalars/angle"] = np.zeros(4)
        f["metadata/fps"] = float(RATE)
    rates = []

    def preview(path, frames, *, fps, **kwargs):
        rates.append(fps)

    monkeypatch.setattr(widget, "write_frames_preview", preview)
    monkeypatch.setattr(widget, "get_flips", lambda frames, **kwargs: np.zeros(len(frames), dtype=bool))
    monkeypatch.setattr(widget, "_note_flip_correction", lambda *args, **kwargs: None)
    flip = widget.FlipClassifierWidget.__new__(widget.FlipClassifierWidget)
    flip.sessions = {"s": h5}
    flip.apply_flip_classifier("clf.p", write_movie=True, verbose=False)
    assert rates == [RATE]
