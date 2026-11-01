"""Write the versions a CI suite job passed with as a pip constraints file.

Usage: python -m pip freeze | python write_constraints.py OUTPUT

Each suite job in tests.yml runs this, and the release workflow attaches the
files from the release pull request's run to the GitHub release, one per
operating system and Python. A user whose fresh install breaks after an
upstream release can then install exactly what CI tested:

    python -m pip install moseq2-app==<version> -c <file>

The five MoSeq2 packages come from the checkout in CI, so the freeze lists
them as local paths. The file pins each at its installed version instead,
which is the version the release publishes. Every other line must be an
exact ``name==version`` pin; anything else stops here, naming the line,
rather than producing a file pip would refuse.
"""

import os
import re
import sys
from importlib.metadata import version
from pathlib import Path

PACKAGES = ("moseq2-app", "moseq2-extract", "moseq2-pca", "moseq2-model", "moseq2-viz")

PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$")
NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def constraints(freeze):
    """The constraint lines for a pip freeze: everything but the five, pinned."""
    lines = []
    for raw in freeze.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # An editable install, as uv prints it, carries a path and no name.
        # In this repository only the five packages are ever installed so.
        if line.startswith("-e "):
            if "moseq2" in line:
                continue
            raise ValueError(line)
        name = NAME.match(line)
        if name and canonical(name.group(1)) in PACKAGES:
            continue
        if not PIN.match(line):
            raise ValueError(line)
        lines.append(line)
    lines.extend(f"{name}=={version(name)}" for name in PACKAGES)
    return lines


def header():
    app = version("moseq2-app")
    lines = [
        f"# The versions moseq2-app {app} passed the CI suite with, on {sys.platform},",
        f"# Python {sys.version_info.major}.{sys.version_info.minor}. To install exactly these:",
        f"#   python -m pip install moseq2-app=={app} -c <this file>",
    ]
    run = os.environ.get("GITHUB_RUN_ID")
    if run:
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        repo = os.environ.get("GITHUB_REPOSITORY", "dattalab/moseq2-app")
        lines.append(f"# From {server}/{repo}/actions/runs/{run}")
    return lines


def main(output):
    try:
        body = constraints(sys.stdin.read())
    except ValueError as err:
        sys.exit(
            f"pip freeze listed {str(err)!r}, which is not an exact name==version pin, "
            f"so it cannot go in the constraints file {output}. A user could not install "
            "it from that line. Install that dependency from a package index in this job, "
            "or have this script handle the line."
        )
    Path(output).write_text("\n".join(header() + body) + "\n")
    print(f"Wrote {len(body)} pins to {output}")


if __name__ == "__main__":
    main(sys.argv[1])
