# Contributing to MoSeq2

This package is one of the five that make up MoSeq2. How changes get merged,
how to run the test suite, which Python versions are tested, what the tests
check, how the packages are versioned and how a release is published are the
same for all five and are documented once:

**https://github.com/dattalab/moseq2-app/blob/main/CONTRIBUTING.md**

Two things are worth knowing before you start here.

This package has no test suite of its own. `moseq2-app`'s suite walks both
notebooks cell by cell and every cell runs through this package, so that is
the suite that can say whether a change here breaks the pipeline. A pull
request runs it automatically, with your checkout laid over the released rest
of the pipeline. To run it yourself, from a `moseq2-app` checkout:

```bash
MOSEQ2_SIBLING_PATHS=/path/to/this/checkout uvx nox -R -s tests-3.10
```

All changes reach `main` through a pull request; direct pushes are blocked and
a pull request merges once its `tests` check passes.
