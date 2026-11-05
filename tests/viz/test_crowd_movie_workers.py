"""Making crowd movies leaves the process able to do linear algebra.

Crowd movies are rendered by a pool of worker processes. Started by forking,
each worker is a copy of the process that asked for the movies: on Linux that
is the notebook's kernel, with JAX's and OpenBLAS's threads running in it.
The OpenBLAS that SciPy 1.17 bundles (0.3.30, the SciPy a Python 3.11 install
resolves to) does not survive a fork: the parent's next multithreaded LAPACK
call waits forever on a lock the fork left taken, holding the GIL. In the
notebook that is a kernel that stops answering, with no error, the next time
anything in it reaches SciPy's linear algebra, such as scikit-learn's PCA in
the flip-classifier training or the fingerprint classifier.

The workers are started fresh instead, as they already are on macOS and
Windows, so the kernel is never copied.
"""

from __future__ import annotations

import multiprocessing.pool
import subprocess
import sys
import textwrap

import pytest

from moseq2_viz.helpers.wrappers import make_crowd_movies_wrapper

#: Long enough for one crowd movie of three examples and a fresh worker to
#: import moseq2-viz on a slow disk; a hang never finishes.
SECONDS = 300

#: The options ``moseq2-viz make-crowd-movies`` passes when given only
#: ``--specific-syllable 0``, ``--max-examples 3`` and ``--processes 2``.
ONE_SYLLABLE = [
    "--specific-syllable", "0", "--max-examples", "3", "--processes", "2",
]


class Refused(Exception):
    """Raised in place of starting the pool, once its start method is seen."""


def test_crowd_movie_workers_are_not_forked(analysis_project_1_6, tmp_path, monkeypatch):
    """The pool is asked for with a start method that does not copy the caller."""
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project
    from moseq2_viz.cli import make_crowd_movies

    seen = []

    def record(self, processes=None, *args, context=None, **kwargs):
        seen.append((context or multiprocessing.get_context()).get_start_method())
        # a pool that never started, so that it has nothing to close when collected
        self._state = multiprocessing.pool.CLOSE
        raise Refused

    monkeypatch.setattr(multiprocessing.pool.Pool, "__init__", record)
    # click fills in every option's default, which is what the wrapper expects
    project = analysis_project_1_6
    config = make_crowd_movies.make_context(
        "make-crowd-movies",
        [str(project / "models" / "model.p"), "--project-file", str(project / "moseq2-project.yaml"),
         *ONE_SYLLABLE],
    ).params
    config.pop("project_file")
    with pytest.raises(Refused):
        make_crowd_movies_wrapper(
            session_index(Project.at(project)), config.pop("model_path"), str(tmp_path), config
        )
    assert seen and "fork" not in seen, f"crowd movie workers were started with {seen}"


def test_linear_algebra_still_answers_after_crowd_movies(analysis_project_1_6, tmp_path):
    """After the movies are written, a LAPACK call large enough to use threads returns.

    Run in its own interpreter: a hang here would otherwise take the whole
    test session with it, past any per-test timeout, since the blocked call
    holds the GIL.
    """
    script = tmp_path / "crowd_then_lu.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import numpy as np
            from scipy.linalg import lu

            from moseq2_viz.cli import make_crowd_movies

            if __name__ == "__main__":
                make_crowd_movies.main(
                    {[str(analysis_project_1_6 / "models" / "model.p"),
                      "--project-file", str(analysis_project_1_6 / "moseq2-project.yaml"),
                      "-o", str(tmp_path / "crowd_movies"), *ONE_SYLLABLE]!r},
                    standalone_mode=False,
                )
                print("crowd movies written", flush=True)
                lu(np.random.default_rng(0).random((400, 400)))
                print("linear algebra answered", flush=True)
            """
        )
    )
    try:
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=SECONDS
        )
    except subprocess.TimeoutExpired as hung:
        out = hung.stdout.decode() if isinstance(hung.stdout, bytes) else hung.stdout
        pytest.fail(
            f"still running after {SECONDS} s; it printed:\n{out}\n"
            "A crowd movie pool started by forking leaves SciPy's OpenBLAS locked."
        )
    assert result.returncode == 0, result.stderr
    assert "linear algebra answered" in result.stdout
    assert list((tmp_path / "crowd_movies").glob("*.mp4")), result.stdout
