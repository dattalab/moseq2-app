# MoSeq2-model 

[![codecov](https://codecov.io/gh/dattalab/moseq2_model/branch/master/graph/badge.svg?token=q9xxVhps5o)](https://codecov.io/gh/dattalab/moseq2_model)
[![DOI](https://zenodo.org/badge/81390468.svg)](https://zenodo.org/badge/latestdoi/81390468)

# Installation

```bash
pip install moseq2-model
```

Python 3.10 or newer, on Linux. macOS and Windows support is being verified.

Most people install the whole pipeline in one step with `pip install
moseq2-app`, which brings this package with it. See [the installation
guide](https://github.com/dattalab/moseq2-app/wiki/MoSeq2-Installation).

# [Documentation: MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki)
You can find more information about MoSeq Pipeline, installation, step-by-step instructions, documentation for Command Line Interface(CLI), tutorials etc in [MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki).

You can run `moseq2-model --version` to check the current version and `moseq2-model --help` to see all the commands.
```bash
Usage: moseq2-model [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.  [default: False]
  --help     Show this message and exit.  [default: False]

Commands:
  count-frames  Counts number of frames in given h5 file (pca_scores)
  kappa-scan    Batch fit multiple models scanning over different syllable...
  learn-model   Trains ARHMM on PCA Scores with given training parameters
```

# Community Support and Contributing
- Please join [![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com) to post questions and interactive with MoSeq developers and users.
- If you encounter bugs, errors or issues, please submit a Bug report [here](https://github.com/dattalab/moseq2-app/issues/new/choose). We encourage you to check out the [troubleshooting and tips section](https://github.com/dattalab/moseq2-app/wiki/Troubleshooting-and-Tips) and search your issues in [the existing issues](https://github.com/dattalab/moseq2-app/issues) first.   
- If you want to see certain features in MoSeq or you have new ideas, please submit a Feature request [here](https://github.com/dattalab/moseq2-app/issues/new/choose).
- If you want to contribute to our codebases, please check out our [Developer Guidelines](https://github.com/dattalab/moseq2-app/wiki/MoSeq-Developer-Guidelines).
- Please tell us what you think by filling out [this user survey](https://forms.gle/FbtEN8E382y8jF3p6).

# License
MoSeq is freely available for academic use under a license provided by Harvard University. Please refer to the license file for details. If you are interested in using MoSeq for commercial purposes please contact Bob Datta directly at srdatta@hms.harvard.edu, who will put you in touch with the appropriate people in the Harvard Technology Transfer office.
