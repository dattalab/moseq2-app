"""Nox sessions for moseq2-app.

Run the suite on one interpreter::

    uvx nox -s tests-3.10

or on every interpreter the package claims to support::

    uvx nox -s tests

moseq2-app depends on four sibling packages. Where those are published, the
install below resolves them normally. Where they are not, point
``MOSEQ2_SIBLING_PATHS`` at local checkouts and they are installed from source
first; the requirement is then already satisfied and nothing reaches for an
index::

    MOSEQ2_SIBLING_PATHS=/path/to/moseq2-extract:/path/to/moseq2-pca uvx nox -s tests
"""

import os

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["tests"]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]


def _install_siblings(session):
    paths = [p for p in os.environ.get("MOSEQ2_SIBLING_PATHS", "").split(os.pathsep) if p.strip()]
    if not paths:
        return
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        session.error(f"MOSEQ2_SIBLING_PATHS names directories that do not exist: {missing}")
    session.install(*paths)


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Install the package as a user would, then run the suite against it."""
    _install_siblings(session)
    session.install(".[tests]")
    # What the resolver actually gave us, so a green run is re-runnable.
    session.run("uv", "pip", "freeze", external=True)
    session.run("pytest", *session.posargs, env={"MPLBACKEND": "Agg"})


@nox.session(python=PYTHON_VERSIONS)
def install(session):
    """Install the package and confirm it imports and answers."""
    _install_siblings(session)
    session.install(".")
    session.run("python", "-c", "import moseq2_app; print(moseq2_app.__version__)")
