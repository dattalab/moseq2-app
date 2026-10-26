"""Cell 1: what is installed.

    !moseq2-extract --version
    !moseq2-pca --version
    !moseq2-model --version
    !moseq2-viz --version

    import sys
    import moseq2_app

    print('MoSeq2 app version:', moseq2_app.__version__)
    print('Python path:', sys.executable)

The four shell lines are the notebook's first contact with the installed
stack and the only place it depends on console scripts being on ``PATH``.
pip writes those scripts into the environment's ``bin/`` and never touches
``PATH``; activating the environment puts that ``bin/`` first and writes
nothing. Both halves are needed, and a cell that reports ``command not
found`` here is almost always a launch that skipped the second one.

The commands are invoked by bare name, not by resolved path, so the test
resolves them the same way the notebook's shell does.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from importlib.metadata import distribution, version

import pytest
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

import moseq2_app

#: The four sibling packages the notebook asks to identify themselves. The
#: console script and the distribution share a name in every case.
SIBLINGS = ("moseq2-extract", "moseq2-pca", "moseq2-model", "moseq2-viz")


def parse_version(text):
    """Pull the version out of a click ``--version`` line.

    click prints ``<prog>, version <X>``, so take the first whitespace- or
    comma-separated token that parses as a version. Returns None if nothing
    in the line does.
    """
    for token in re.split(r"[\s,]+", text.strip()):
        try:
            return Version(token)
        except InvalidVersion:
            continue
    return None


@pytest.fixture(scope="module")
def cli_version_output():
    """Run ``<cli> --version`` once per sibling, by bare name through PATH.

    Returns a name -> CompletedProcess mapping, with None where the console
    script is not on PATH at all, so that case fails as a test rather than
    as an error raised out of this fixture.
    """
    results = {}
    for name in SIBLINGS:
        if shutil.which(name) is None:
            results[name] = None
            continue
        results[name] = subprocess.run(
            [name, "--version"], capture_output=True, text=True, timeout=300
        )
    return results


@pytest.fixture(scope="module")
def declared_specifiers():
    """What moseq2-app's own metadata accepts for each sibling.

    Read from the installed metadata rather than restated here, so the test
    tracks the requirement the package actually ships.
    """
    found = {}
    for raw in distribution("moseq2-app").requires or []:
        req = Requirement(raw)
        if req.name in SIBLINGS:
            found[req.name] = req.specifier
    missing = set(SIBLINGS) - set(found)
    assert not missing, f"moseq2-app's metadata does not require: {sorted(missing)}"
    return found


@pytest.mark.parametrize("cli", SIBLINGS)
def test_console_script_is_on_path_and_answers(cli, cli_version_output):
    """1. The command the notebook shells out to exists and exits cleanly."""
    proc = cli_version_output[cli]
    assert proc is not None, (
        f"`{cli}` is not on PATH. pip installs it into the environment's bin/; "
        "activating the environment is what puts that bin/ on PATH."
    )
    assert proc.returncode == 0, (
        f"`{cli} --version` exited {proc.returncode}\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )


@pytest.mark.parametrize("cli", SIBLINGS)
def test_console_script_reports_a_version(cli, cli_version_output):
    """2. What it printed is a version, not just any output."""
    proc = cli_version_output[cli]
    assert proc is not None and proc.returncode == 0, f"`{cli} --version` did not run"
    reported = parse_version(proc.stdout)
    assert reported is not None, f"no version in `{cli} --version` output: {proc.stdout!r}"


@pytest.mark.parametrize("cli", SIBLINGS)
def test_installed_sibling_satisfies_the_app_requirement(cli, cli_version_output, declared_specifiers):
    """3. The installed sibling is one moseq2-app declares it works with."""
    proc = cli_version_output[cli]
    assert proc is not None and proc.returncode == 0, f"`{cli} --version` did not run"
    reported = parse_version(proc.stdout)
    assert reported is not None, f"no version in `{cli} --version` output: {proc.stdout!r}"
    specifier = declared_specifiers[cli]
    assert specifier.contains(reported), (
        f"{cli} {reported} does not satisfy moseq2-app's requirement "
        f"{cli}{specifier}"
    )


@pytest.mark.parametrize("cli", SIBLINGS)
def test_console_script_agrees_with_installed_distribution(cli, cli_version_output):
    """4. The script on PATH belongs to the distribution that is installed.

    These disagree when a console script left behind by an earlier install
    is still first on PATH. Nothing else in the cell would notice.
    """
    proc = cli_version_output[cli]
    assert proc is not None and proc.returncode == 0, f"`{cli} --version` did not run"
    reported = parse_version(proc.stdout)
    installed = Version(version(cli))
    assert reported == installed, (
        f"`{cli}` on PATH reports {reported} but the installed {cli} "
        f"distribution is {installed}; the script resolves to a different install "
        f"({shutil.which(cli)})"
    )


def test_app_reports_its_own_version():
    """5. The attribute the cell prints exists, parses, and is not a lie.

    moseq2-app takes its packaged version from ``moseq2_app/__init__.py``, so
    the attribute and the distribution metadata are the same number read two
    ways. If they disagree, the cell prints something untrue about what is
    installed.
    """
    assert hasattr(moseq2_app, "__version__"), "moseq2_app has no __version__"
    attribute = Version(moseq2_app.__version__)
    packaged = Version(version("moseq2-app"))
    assert attribute == packaged, (
        f"moseq2_app.__version__ is {attribute} but the installed distribution "
        f"is {packaged}"
    )
