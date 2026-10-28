"""Nox sessions for the MoSeq2 pipeline.

Run the suite on one interpreter::

    uvx nox -s tests-3.10

or on every interpreter the packages claim to support::

    uvx nox -s tests

Every session installs all five packages from this checkout: moseq2-app from
the root and the other four from packages/. They go in one resolution, so the
four checkouts satisfy moseq2-app's exact pins on them and each brings its
own requirements with it.
"""

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["tests"]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

SIBLINGS = [
    "packages/moseq2-extract",
    "packages/moseq2-pca",
    "packages/moseq2-model",
    "packages/moseq2-viz",
]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Install the pipeline as a user would, then run the suite against it."""
    session.install(*SIBLINGS, ".[tests]")
    # What the resolver actually gave us, so a green run is re-runnable.
    session.run("uv", "pip", "freeze", external=True)
    session.run("pytest", *session.posargs, env={"MPLBACKEND": "Agg"})


@nox.session(python=PYTHON_VERSIONS)
def install(session):
    """Install the pipeline and confirm it imports and answers."""
    session.install(*SIBLINGS, ".")
    session.run("python", "-c", "import moseq2_app; print(moseq2_app.__version__)")
