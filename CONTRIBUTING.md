# Contributing to MoSeq2

MoSeq2 is five packages: `moseq2-app` and the four it drives, `moseq2-extract`,
`moseq2-pca`, `moseq2-model` and `moseq2-viz`. All five live in this
repository, `moseq2-app` at the root and the other four under `packages/`, and
each is published to PyPI under its own name. This page covers how changes get
into any of them and how to run and extend the test suite.

## How changes get merged

All changes reach `main` through a pull request; direct pushes are blocked. A
pull request can be merged once its `tests` check passes, by anyone with write
access, including its own author. No second approval is required, and nobody
can merge a pull request whose check is red.

`main` is where work lands. `release` points at the newest published version
and moves only when a release is published, so the notebook links in the
documentation always open the version a user can install.

## Running the tests

The suite runs the steps of both notebooks against a small fixture dataset,
and every step runs through the four packages under `packages/`, so it is the
test for all five. With [uv](https://docs.astral.sh/uv/) installed, run it from
a checkout of this repository:

```bash
uvx nox -s tests-3.10
```

This builds a fresh environment, installs all five packages from the checkout
along with `moseq2-app`'s `tests` dependencies, and runs pytest. A change to
any of the five is therefore tested by the same command. The first run downloads about 640 MB of
fixtures and caches them in your user cache directory. The suite takes roughly
eight minutes.

```bash
uvx nox -s tests-3.10 tests-3.14          # the versions a pull request tests
uvx nox -R -s tests-3.10                  # reuse the existing environment
uvx nox -R -s tests-3.10 -- tests/analysis/test_07_export_stats_df.py
uvx nox -R -s tests-3.10 -- -k uuid       # tests whose names match
```

## Which Python versions are tested

MoSeq2 supports Python 3.10 through 3.14. The list lives in `PYTHON_VERSIONS`
in `noxfile.py`. Every run installs the newest release of each dependency the
requirements allow, and those differ by Python version: 3.10 gets pandas 2 and
an older NumPy, later versions get pandas 3.

| When | Python versions | Required to merge |
|---|---|---|
| Every push to a pull request | 3.10 and 3.14 | Yes |
| Nightly, and on demand | 3.10 through 3.14 | No |

A pull request tests the oldest and newest because those bracket the range of
dependency releases. All three operating systems run, because the suite exists
to answer whether a user on one gets what a user on another gets. Each cell is
its own job and they run in parallel, and the `tests` check passes only when
every one of them did.

## What the tests check

Most tests record what MoSeq2 does today, not what it should do. Their job is
to catch changes nobody intended, so a passing test is not an endorsement of
the behaviour it checks. A failing test means behaviour changed; whether that
is a problem depends on whether the change was intended.

Known bugs are recorded the same way. A test marked `xfail` reproduces a
defect that exists today, and its `reason` names it. `xfail_strict` is on, so
when a change fixes the bug that test fails because the bug no longer happens:
remove the marker and make the test check the corrected behaviour.

A test that hangs is reported as a hang. `pytest.ini` sets a per-test timeout,
so a test that stops making progress fails in ten minutes with the stack of
every thread rather than running until the job times out with nothing to read.

## Version numbers

The five packages share one version and release together. `moseq2-app` pins
the other four at exactly its own version, so installing `moseq2-app` at a
version installs the whole pipeline at that version.

Each notebook records the minor version it is written for in its notebook
metadata, as `"moseq2": {"version": "1.4"}`. A patch release leaves it alone.

A minor release either substantially changes a result or changes something
about the notebook interface: which cells a user runs, what a cell takes, or
what it hands to the next one. New features are minor releases too.

Everything else is a patch: crash fixes, documentation, performance
improvements, tightened requirements that exclude a broken upstream release,
and a fix whose effect on results is incidental rather than substantial. A fix
that replaces no result at all with a correct one is a patch, because nobody
had a result to compare against.

If your pull request changes outputs, say so in its description, and say
whether you think the change is substantial.

## Releasing

Publish a GitHub release against a tag on `main`. The release workflow builds
all five packages from the tag and uploads them to PyPI over trusted
publishing, so no token is stored anywhere. Before uploading anything it
checks that every package declares the version the tag names, that
`moseq2-app` pins the other four at that version, and that every notebook
names the tag's minor version. It uploads `moseq2-app` last, so a failure
partway never leaves it requiring a release that is not there. After
uploading it waits for the index, then moves `release` to the tag.

Bump the version in its own pull request first: `__version__` in all five
`__init__.py` files and the four pins in `moseq2-app`'s `pyproject.toml`, plus
the notebooks' metadata for a minor release. A tag cut before that bump
publishes the previous release's number.

## Reporting problems

Open an issue on this repository. Include the version of each MoSeq2 package
you have, your Python version, your operating system, and the full error
message.
