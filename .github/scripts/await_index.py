"""Wait until PyPI's index serves a version, or give up.

Usage: python await_index.py NAME VERSION [TIMEOUT_SECONDS]

An upload returns before the index that installers read has caught up, so a
run that publishes a package and then installs it can resolve the previous
release. Waiting here keeps that race out of every downstream workflow.
"""

import sys
import time
import urllib.request


def served(name, version):
    url = f"https://pypi.org/simple/{name}/"
    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return f"-{version}" in response.read().decode()


def main(name, version, timeout=600):
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if served(name, version):
            print(f"{name} {version} is on the index")
            return
        time.sleep(10)
    sys.exit(f"{name} {version} did not reach the index within {timeout}s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
