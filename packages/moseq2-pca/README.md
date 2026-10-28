# MoSeq2-PCA

[![Build Status](https://travis-ci.com/dattalab/moseq2-pca.svg?token=gvoikVySDHEmvHT7Dbed&branch=test-suite)](https://travis-ci.com/dattalab/moseq2-pca) 

[![codecov](https://codecov.io/gh/dattalab/moseq2-pca/branch/test-suite/graph/badge.svg?token=OLbqEbHHNP)](https://codecov.io/gh/dattalab/moseq2-pca)

This is a library for computing PCA components and scores from extracted mouse movies.  Use this to compute features for modeling.

# Installation

`moseq2-pca` is one package of the MoSeq2 pipeline. Install the whole pipeline:

```bash
python -m pip install moseq2-app
```

You need Python 3.10 or newer on your system. [The installation
guide](https://github.com/dattalab/moseq2-app/wiki/MoSeq2-Installation) carries
the whole procedure, including the notebooks.

To install this package on its own: `python -m pip install moseq2-pca`.

# [Documentation: MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki)
You can find more information about MoSeq Pipeline, installation, step-by-step instructions, documentation for Command Line Interface(CLI), tutorials etc in [MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki).

You can run `moseq2-pca --version` to check the current version and `moseq2-pca --help` to see all the commands.
```bash
Usage: moseq2-pca [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.  [default: False]
  --help     Show this message and exit.  [default: False]

Commands:
  apply-pca             Computes PCA Scores of extraction data given a...
  clip-scores           Clips specified number of frames from PCA scores at...
  compute-changepoints  Computes the Model-Free Syllable Changepoints based...
  train-pca             Trains PCA on all extracted results (h5 files) in...

```

# Community Support and Contributing
- Please join [![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com) to post questions and interactive with MoSeq developers and users.
- If you encounter bugs, errors or issues, please submit a Bug report [here](https://github.com/dattalab/moseq2-app/issues/new/choose). We encourage you to check out the [troubleshooting and tips section](https://github.com/dattalab/moseq2-app/wiki/Troubleshooting-and-Tips) and search your issues in [the existing issues](https://github.com/dattalab/moseq2-app/issues) first.   
- If you want to see certain features in MoSeq or you have new ideas, please submit a Feature request [here](https://github.com/dattalab/moseq2-app/issues/new/choose).
- If you want to contribute to our codebases, please check out our [Developer Guidelines](https://github.com/dattalab/moseq2-app/wiki/MoSeq-Developer-Guidelines).
- Please tell us what you think by filling out [this user survey](https://forms.gle/FbtEN8E382y8jF3p6).

# License

MoSeq is freely available for academic use under a license provided by Harvard University. Please refer to the license file for details. If you are interested in using MoSeq for commercial purposes please contact Bob Datta directly at srdatta@hms.harvard.edu, who will put you in touch with the appropriate people in the Harvard Technology Transfer office.

