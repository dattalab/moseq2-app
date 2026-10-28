"""Refuse a release whose metadata disagrees with the tag it was cut from.

Usage: python check_release.py TAG PACKAGE

Checks two things and exits non-zero on either:

* ``PACKAGE/__init__.py`` declares the version the tag names. A tag cut
  before the version bump publishes the previous release's number, which
  PyPI then refuses as a duplicate or, worse, accepts as a re-release of
  content that is not what the tag says.
* Every moseq2 sibling this package requires exists on PyPI at a version the
  requirement admits. A floor naming a sibling release that was never
  published installs nothing, and the failure surfaces in whatever
  downstream run tries it rather than here.
"""

import json
import pathlib
import re
import sys
import urllib.request

from packaging.requirements import Requirement
from packaging.version import Version


def declared_version(package):
    text = (pathlib.Path(package) / "__init__.py").read_text()
    match = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if match is None:
        sys.exit(f"{package}/__init__.py declares no __version__")
    return match.group(1)


def released_versions(name):
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return list(json.load(response)["releases"])


def siblings_of(package):
    """The moseq2 requirements this package declares, from its own metadata.

    A requirement carries a specifier; the package's own name, which appears
    in the same files, does not. That is what separates the two.
    """
    seen = set()
    for path in (pathlib.Path("pyproject.toml"), pathlib.Path("setup.py")):
        if not path.exists():
            continue
        text = path.read_text()
        for raw in re.findall(r"""["'](moseq2-[a-z]+[^"']*)["']""", text):
            try:
                requirement = Requirement(raw)
            except Exception:
                # Console scripts live in these files too, and
                # "moseq2-viz = moseq2_viz.cli:cli" is not a requirement.
                continue
            if not requirement.specifier or requirement.name in seen:
                continue
            seen.add(requirement.name)
            yield requirement


def main(tag, package):
    wanted = tag.lstrip("v")
    declared = declared_version(package)
    if Version(declared) != Version(wanted):
        sys.exit(f"tag {tag} names {wanted}, but {package} declares {declared}")
    print(f"{package} declares {declared}, matching {tag}")

    for requirement in siblings_of(package):
        available = [v for v in released_versions(requirement.name)
                     if requirement.specifier.contains(v)]
        if not available:
            sys.exit(
                f"{requirement} has no release on PyPI that satisfies it; "
                "publish that sibling first"
            )
        print(f"{requirement} is satisfied by {max(available, key=Version)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
