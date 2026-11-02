"""Refuse a release whose metadata disagrees with the tag it was cut from.

Usage: python check_release.py TAG

The six packages release together, at one version and from one tag:
moseq2-app at the root and the other five under packages/. Checks four
things, reports every disagreement it finds, and exits non-zero on any:

* Every package's ``__init__.py`` declares the version the tag names. A tag
  cut before the version bump publishes the previous release's number, which
  PyPI then refuses as a duplicate, and a bump that missed one package
  publishes a set whose numbers disagree.
* moseq2-app pins each of the other five at exactly that version. The same
  run builds and uploads all six, so a pin naming anything else names a
  release this one does not make.
* Each of the other five requires moseq2-common at exactly that version,
  for the same reason: the same run uploads it, so a pin naming anything else
  names a release this run does not make.
* Every notebook carries the release's major.minor in its metadata, under
  ``moseq2.version``. A notebook is written for a minor release, since a
  patch changes nothing about which cells a user runs or what they hand on.
  A notebook without the key fails too, which catches an editor that drops
  metadata it does not recognise.
"""

import json
import pathlib
import re
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


def packages():
    """Each package's distribution name and the directory its pyproject is in."""
    found = {"moseq2-app": pathlib.Path(".")}
    for directory in sorted(pathlib.Path("packages").iterdir()):
        if (directory / "pyproject.toml").exists():
            found[directory.name] = directory
    return found


def declared_version(name, directory):
    init = directory / name.replace("-", "_") / "__init__.py"
    match = re.search(r'^__version__ = "([^"]+)"', init.read_text(), re.M)
    return None if match is None else match.group(1)


def moseq2_requirements(directory):
    """The moseq2-* requirements in ``directory/pyproject.toml``, by name."""
    with open(directory / "pyproject.toml", "rb") as f:
        dependencies = tomllib.load(f)["project"].get("dependencies", [])
    requirements = (Requirement(d) for d in dependencies)
    return {r.name: r for r in requirements if r.name.startswith("moseq2-")}


def notebook_version(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["metadata"].get("moseq2", {}).get("version")


def main(tag):
    version = Version(tag.lstrip("v"))
    minor = f"{version.major}.{version.minor}"
    problems = []

    found = packages()
    for name, directory in found.items():
        declared = declared_version(name, directory)
        if declared is None or Version(declared) != version:
            problems.append(f"{name} declares {declared}, but {tag} names {version}")
        else:
            print(f"{name} declares {declared}")

    pins = moseq2_requirements(pathlib.Path("."))
    for name in sorted(set(found) - {"moseq2-app"}):
        pin = pins.get(name)
        if pin is None:
            problems.append(f"moseq2-app does not require {name}")
        elif str(pin.specifier) != f"=={version}":
            problems.append(f"moseq2-app requires {pin}, not {name}=={version}")
        else:
            print(f"moseq2-app requires {pin}")
    for name in sorted(set(pins) - set(found)):
        problems.append(f"moseq2-app requires {pins[name]}, which this repository does not build")

    # moseq2-app's own moseq2-common pin is among the pins checked above.
    for name, directory in found.items():
        if name in ("moseq2-app", "moseq2-common"):
            continue
        pin = moseq2_requirements(directory).get("moseq2-common")
        if pin is None:
            problems.append(f"{name} does not require moseq2-common")
        elif str(pin.specifier) != f"=={version}":
            problems.append(f"{name} requires {pin}, not moseq2-common=={version}")
        else:
            print(f"{name} requires {pin}")

    notebooks = sorted(pathlib.Path("notebooks").glob("*.ipynb"))
    if not notebooks:
        problems.append("no notebooks found under notebooks/")
    for path in notebooks:
        declared = notebook_version(path)
        if declared is None:
            problems.append(f"{path} carries no moseq2.version in its metadata")
        elif declared != minor:
            problems.append(f"{path} is written for {declared}, not {minor}")
        else:
            print(f"{path} is written for {declared}")

    if problems:
        sys.exit("\n".join(["The release disagrees with its tag:", *problems]))


if __name__ == "__main__":
    main(sys.argv[1])
