"""Fetch the recordings and the model bundle the suite runs against.

Two bundles, because the two notebooks need different things.

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
