"""The constraints file each CI suite job writes for the release.

Every suite job hands its ``pip freeze`` to ``.github/scripts/write_constraints.py``,
which writes the versions that job passed with as a pip constraints file. The
release workflow attaches one per operating system and Python to the GitHub
release, so a user whose fresh install breaks can install exactly what CI
tested with ``python -m pip install moseq2-app==<version> -c <file>``.

The five MoSeq2 packages appear in a CI freeze as local paths (the checkout
they were installed from), which are useless to a user, so the file pins each
at its released version instead. Everything else is pinned exactly as the
freeze reports it, and a line that cannot be pinned stops the job rather than
shipping a file pip would refuse.

The script lives in the repository, not the sdist, so these tests skip when
run from an unpacked sdist.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "write_constraints.py"
PACKAGES = ("moseq2-app", "moseq2-extract", "moseq2-pca", "moseq2-model", "moseq2-viz")

pytestmark = pytest.mark.skipif(
    not (REPO / ".github").is_dir(), reason="the release scripts are not part of the sdist"
)

#: What pip freeze printed in a CI suite job, cut down: the five packages from
#: the checkout, as pip and uv print local and editable installs, among
#: ordinary pins.
FREEZE = """\
av==17.1.0
-e file:///home/runner/work/moseq2-app/moseq2-app/packages/moseq2-model
moseq2-app @ file:///home/runner/work/moseq2-app/moseq2-app
moseq2-extract @ file:///home/runner/work/moseq2-app/moseq2-app/packages/moseq2-extract
moseq2_pca @ file:///D:/a/moseq2-app/moseq2-app/packages/moseq2-pca
moseq2-viz==1.4.5
numpy==2.2.6\r
opencv-python-headless==5.0.0.93

pywin32==311
"""


def write(tmp_path, freeze):
    out = tmp_path / "constraints-linux-py3.10.txt"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(out)],
        input=freeze,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc, out


def pins(path):
    """name -> version for every requirement line in a constraints file."""
    found = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        req = Requirement(line)
        (spec,) = req.specifier
        assert spec.operator == "==", f"not an exact pin: {line!r}"
        found[req.name] = spec.version
    return found


def test_the_script_exists():
    assert SCRIPT.is_file(), f"{SCRIPT} is missing; tests.yml runs it in every suite job"


def test_moseq2_packages_are_pinned_at_their_installed_versions(tmp_path):
    proc, out = write(tmp_path, FREEZE)
    assert proc.returncode == 0, proc.stderr
    found = pins(out)
    for package in PACKAGES:
        assert found.get(package) == version(package), (package, found)
    assert not any(n.replace("_", "-") in PACKAGES and n not in PACKAGES for n in found)


def test_every_other_pin_is_kept_as_frozen(tmp_path):
    proc, out = write(tmp_path, FREEZE)
    assert proc.returncode == 0, proc.stderr
    found = pins(out)
    others = {k: v for k, v in found.items() if k not in PACKAGES}
    assert others == {
        "av": "17.1.0",
        "numpy": "2.2.6",
        "opencv-python-headless": "5.0.0.93",
        "pywin32": "311",
    }


def test_the_file_says_how_to_use_it(tmp_path):
    proc, out = write(tmp_path, FREEZE)
    assert proc.returncode == 0, proc.stderr
    header = [line for line in out.read_text().splitlines() if line.startswith("#")]
    usage = f"python -m pip install moseq2-app=={version('moseq2-app')} -c"
    assert any(usage in line for line in header), header


def test_a_line_that_is_not_a_pin_stops_with_the_line(tmp_path):
    freeze = FREEZE + "somelib @ git+https://github.com/example/somelib@abc123\n"
    proc, out = write(tmp_path, freeze)
    assert proc.returncode != 0
    assert "somelib @ git+https://github.com/example/somelib@abc123" in proc.stderr
    assert not out.exists()
