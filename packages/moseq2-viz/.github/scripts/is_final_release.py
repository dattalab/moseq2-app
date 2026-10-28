"""Report whether a release tag names a final release.

Usage: python is_final_release.py TAG

The release branch points at the newest release a user can install, and pip
never installs a pre-release by default, so the release workflow moves it
only for a final one. Prints a GitHub Actions step output.
"""

import re
import sys


def is_final(tag):
    return re.fullmatch(r"v?\d+(\.\d+)*", tag) is not None


if __name__ == "__main__":
    print(f"final={str(is_final(sys.argv[1])).lower()}")
