"""Fetch the recordings and the model bundle the suite runs against.

Three bundles. Two of them are what the notebooks need; the third is a
single synthetic recording the video tests read.

``raw-sessions`` is four depth recordings cut down to 600 frames each, two
per experimental group. The extraction and modeling notebook runs on these
and fits a model for a few iterations. Small enough to extract quickly,
structured enough that the index file, the group assignment and every
group-wise comparison downstream are real.

``analysis-bundle`` is a snapshot of a finished project: twenty sessions,
aggregated and PCA'd, and a model fit for 100 iterations over all of them.
The analysis notebook reads it rather than fitting anything. A model fit on
80 seconds of mouse produces syllables that are noise, and a dendrogram or a
transition matrix built on noise says nothing about whether those functions
work.

Both are fetched with pooch into the platform cache directory and are only
downloaded once. ``MOSEQ2_APP_FIXTURE_URL`` points at an http mirror; pooch
speaks http, https, ftp, sftp and doi, so a path on disk is not a mirror.
To run against a bundle that is not served anywhere yet, drop the tarball
into the cache directory under its registered name and pooch will verify its
hash and unpack it rather than reaching for the network.

``azure-mkv`` is one 64x48, 60-frame synthetic Azure Kinect recording:
a ``COLOR`` track and a 16-bit big-endian ``DEPTH`` track, named by Matroska
title tags the way a real one is. No public Azure Kinect sample is
downloadable, and the depth reader is the only part of the video layer that
no real recording in the suite exercises.

``analysis-bundle`` carries no preview movies. The index names each session's
h5 and its yaml sidecar, crowd movies are composited out of the h5 frames,
and the twenty extraction mp4s are 200 MB that nothing in the notebook opens.
"""

from __future__ import annotations

import os
from pathlib import Path

import pooch

#: Where the bundles are served from. Public read, no credentials.
BASE_URL = os.environ.get(
    "MOSEQ2_APP_FIXTURE_URL",
    "https://moseq2-app-fixtures.s3.amazonaws.com/v1/",
)

#: Filename and hash of every bundle. Written by the build script that cuts
#: the fixtures, so the suite always fetches the bundle it was pinned to.
REGISTRY_FILE = Path(__file__).parent / "fixture_registry.txt"

RAW_SESSIONS = "raw-sessions.tar.gz"
ANALYSIS_BUNDLE = "analysis-bundle.tar.gz"
AZURE_MKV = "azure-mkv.tar.gz"


def _registry():
    if not REGISTRY_FILE.exists():
        raise RuntimeError(
            f"{REGISTRY_FILE} is missing, so there is no hash to verify a download "
            "against. It is written by the fixture build script and committed."
        )
    cache = pooch.create(path=pooch.os_cache("moseq2-app-tests"), base_url=BASE_URL, registry=None)
    cache.load_registry(REGISTRY_FILE)
    return cache


def _fetch_bundle(name):
    """Download and unpack one bundle; return the directory it unpacked to.

    Every bundle holds a single top-level directory, so that is what comes back.
    """
    members = _registry().fetch(name, processor=pooch.Untar())
    roots = {Path(m).parent for m in members}
    common = os.path.commonpath([str(r) for r in roots])
    return Path(common)


def fetch_raw_sessions():
    """The four cut-down depth recordings, plus the flip classifier."""
    return _fetch_bundle(RAW_SESSIONS)


def fetch_analysis_bundle():
    """The finished twenty-session project the analysis notebook reads."""
    return _fetch_bundle(ANALYSIS_BUNDLE)


def fetch_azure_mkv():
    """The synthetic Azure Kinect recording, holding ``depth.mkv``."""
    return _fetch_bundle(AZURE_MKV)


#: What the analysis bundle's progress file carries where the project root belongs.
ROOT_SENTINEL = "__PROJECT_ROOT__"


def project_1_6_from_bundle(bundle, dest):
    """A copy of the analysis bundle at ``dest``, laid out as a 1.6 project.

    The bundle is a project made by 1.5.0: a flat ``config.yaml``, a
    ``moseq2-index.yaml`` naming each session's aggregated extraction and group,
    and a ``progress.yaml``. The copy's ``config.yaml`` holds the same settings in
    their stages' sections, with the index's groups as its ``groups``, and its
    ``moseq2-project.yaml`` lists the index's sessions (keyed by ``SessionName``,
    since the bundle holds no recordings), the PCA files and the model. The index
    and progress file stay beside them, with the copy's folder in place of the
    root the progress file was cut with, for the analysis cells that read them.
    """
    import shutil

    from moseq2_common.options import setting_fields
    from moseq2_common.yaml import read_yaml, write_config, write_yaml
    from moseq2_extract.options import EXTRACT_STAGE
    from moseq2_model.options import MODEL_STAGE
    from moseq2_pca.options import PCA_STAGE
    from moseq2_viz.options import CROWD_MOVIE_STAGE

    dest = Path(dest)
    shutil.copytree(bundle, dest, dirs_exist_ok=True)
    progress = dest / "progress.yaml"
    if progress.is_file():
        progress.write_text(progress.read_text().replace(ROOT_SENTINEL, str(dest)))

    flat = read_yaml(str(dest / "config.yaml")) or {}
    config = {"camera_type": flat.get("camera_type", "auto")}
    for section, stage in (("extract", EXTRACT_STAGE), ("pca", PCA_STAGE), ("model", MODEL_STAGE),
                           ("crowd_movies", CROWD_MOVIE_STAGE)):
        names = {name for cls in stage for name in setting_fields(cls)} - {"camera_type"}
        config[section] = {key: value for key, value in flat.items() if key in names}

    index = read_yaml(str(dest / "moseq2-index.yaml"))
    sessions, groups = {}, {}
    for entry in index["files"]:
        key = str(entry["metadata"]["SessionName"])
        sessions[key] = {"recording": key, "folder": Path(entry["path"][0]).as_posix(), "uuid": str(entry["uuid"])}
        groups[key] = str(entry["group"])
    config["groups"] = groups
    write_config(config, str(dest / "config.yaml"))

    outputs = {"components": "_pca/pca.h5", "scores": "_pca/pca_scores.h5", "changepoints": "_pca/changepoints.h5"}
    write_yaml({
        "sessions": sessions,
        "pca": {name: path for name, path in outputs.items() if (dest / path).is_file()},
        "models": {"fitted": ["models/model.p"], "current": "models/model.p", "analysis": {}},
        "applied_models": [],
    }, str(dest / "moseq2-project.yaml"))
    return dest
