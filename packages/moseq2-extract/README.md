# MoSeq2-Extract: Depth Video Rodent-Tracking Toolkit
 
[![Build Status](https://travis-ci.com/dattalab/moseq2-extract.svg?token=gvoikVySDHEmvHT7Dbed&branch=test-suite)](https://travis-ci.com/dattalab/moseq2-extract)
[![codecov](https://codecov.io/gh/dattalab/moseq2-extract/branch/test-suite/graph/badge.svg?token=ICPjpMMwYZ)](https://codecov.io/gh/dattalab/moseq2-extract)
[![DOI](https://zenodo.org/badge/120829989.svg)](https://zenodo.org/badge/latestdoi/120829989)

# Installation

`moseq2-extract` is one package of the MoSeq2 pipeline. Install the whole pipeline:

```bash
python -m pip install moseq2-app
```

You need Python 3.10 or newer on your system. [The installation
guide](https://github.com/dattalab/moseq2-app/wiki/MoSeq2-Installation) carries
the whole procedure, including the notebooks.

To install this package on its own: `python -m pip install moseq2-extract`.

# [Documentation: MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki)
You can find more information about MoSeq Pipeline, installation, step-by-step instructions, documentation for Command Line Interface(CLI), tutorials etc in [MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki).

You can run `moseq2-extract --version` to check the current version and `moseq2-extract --help` to see all the commands.
```bash
Usage: moseq2-extract [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.  [default: False]
  --help     Show this message and exit.  [default: False]

Commands:
  aggregate-results   Copies all extracted results (h5, yaml, avi) files...
  batch-extract       Batch processes all the raw depth recordings located...
  convert-raw-to-avi  Converts/Compresses a raw depth file into an avi file...
  copy-slice          Copies a segment of an input depth recording into a...
  download-flip-file  Downloads Flip-correction model that helps with...
  extract             Processes raw input depth recordings to output a...
  find-roi            Finds the ROI and background distance to subtract
                      from...
  generate-config     Generates a configuration file that holds editable...
  generate-index      Generates an index YAML file containing all extracted...
```


# Community Support and Contributing
- Please join [![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com) to post questions and interactive with MoSeq developers and users.
- If you encounter bugs, errors or issues, please submit a Bug report [here](https://github.com/dattalab/moseq2-app/issues/new/choose). We encourage you to check out the [troubleshooting and tips section](https://github.com/dattalab/moseq2-app/wiki/Troubleshooting-and-Tips) and search your issues in [the existing issues](https://github.com/dattalab/moseq2-app/issues) first.   
- If you want to see certain features in MoSeq or you have new ideas, please submit a Feature request [here](https://github.com/dattalab/moseq2-app/issues/new/choose).
- If you want to contribute to our codebases, please check out our [Developer Guidelines](https://github.com/dattalab/moseq2-app/wiki/MoSeq-Developer-Guidelines).
- Please tell us what you think by filling out [this user survey](https://forms.gle/FbtEN8E382y8jF3p6).

# License

MoSeq is freely available for academic use under a license provided by Harvard University. Please refer to the license file for details. If you are interested in using MoSeq for commercial purposes please contact Bob Datta directly at srdatta@hms.harvard.edu, who will put you in touch with the appropriate people in the Harvard Technology Transfer office.
