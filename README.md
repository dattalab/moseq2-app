# MoSeq2-Viz: Visualization toolbox for MoSeq2

[![Build Status](https://travis-ci.com/dattalab/moseq2-viz.svg?token=XJGe3NpjKFY5oYjdNcmg&branch=test-suite)](https://travis-ci.com/dattalab/moseq2-viz)
[![codecov](https://codecov.io/gh/dattalab/moseq2-viz/branch/test-suite/graph/badge.svg?token=jUx63Whtx4)](https://codecov.io/gh/dattalab/moseq2-viz)
[![DOI](https://zenodo.org/badge/132665227.svg)](https://zenodo.org/badge/latestdoi/132665227)

# Installation

```bash
python -m pip install moseq2-viz
```

Python 3.10 or newer, on Linux. macOS and Windows support is being verified.

Most people install the whole pipeline in one step with `pip install
moseq2-app`, which brings this package with it. See [the installation
guide](https://github.com/dattalab/moseq2-app/wiki/MoSeq2-Installation).

# [Documentation: MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki)
You can find more information about MoSeq Pipeline, installation, step-by-step instructions, documentation for Command Line Interface(CLI), tutorials etc in [MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki).

You can run `moseq2-viz --version` to check the current version and `moseq2-viz --help` to see all the commands.
```bash
Usage: moseq2-viz [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.  [default: False]
  --help     Show this message and exit.  [default: False]

Commands:
  add-group                       Change group name in index file given a...
  copy-h5-metadata-to-yaml        Copies metadata within an h5 file to a...
  get-best-model                  Returns the model with the closest median...
  make-crowd-movies               Writes movies of overlaid examples of the...
  plot-group-position-heatmaps    Plots position heatmaps for each group in...
  plot-scalar-summary             Plots a scalar summary of the index file...
  plot-stats                      Plots syllable usages with different...
  plot-transition-graph           Plots the transition graph depicting the...
  plot-verbose-position-heatmaps  Plots a position heatmap for each session...
```

# Community Support and Contributing
- Please join [![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com) to post questions and interactive with MoSeq developers and users.
- If you encounter bugs, errors or issues, please submit a Bug report [here](https://github.com/dattalab/moseq2-app/issues/new/choose). We encourage you to check out the [troubleshooting and tips section](https://github.com/dattalab/moseq2-app/wiki/Troubleshooting-and-Tips) and search your issues in [the existing issues](https://github.com/dattalab/moseq2-app/issues) first.   
- If you want to see certain features in MoSeq or you have new ideas, please submit a Feature request [here](https://github.com/dattalab/moseq2-app/issues/new/choose).
- If you want to contribute to our codebases, please check out our [Developer Guidelines](https://github.com/dattalab/moseq2-app/wiki/MoSeq-Developer-Guidelines).
- Please tell us what you think by filling out [this user survey](https://forms.gle/FbtEN8E382y8jF3p6).

# License
MoSeq is freely available for academic use under a license provided by Harvard University. Please refer to the license file for details. If you are interested in using MoSeq for commercial purposes please contact Bob Datta directly at srdatta@hms.harvard.edu, who will put you in touch with the appropriate people in the Harvard Technology Transfer office.
