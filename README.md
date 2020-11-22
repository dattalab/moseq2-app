# MoSeq2-Notebook: A suite of interactive Jupyter notebooks for animal behavior sequencing and analysis

[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/dattalab/community)

<center><img src="https://drive.google.com/uc?export=view&id=1jCpb4AQKBasne2SoXw_S1kwU7R9esAF6"></center>

Last Updated: 10/12/2020

## Overview

The MoSeq2 toolkit enables users to model rodent behavior across different experimental groups, and
measure the differences between their behavior usages, durations, transition patterns. etc.

This package contains interactive jupyter notebooks that are tailored for novice programmers to process
their depth videos of rodents, and segment their behavior into what is denoted as "syllables".

Consult the wiki-page for a complete overview and description of the MoSeq pipeline [here](https://github.com/dattalab/moseq2-app/wiki).

__Note: The Notebook is currently in a testing phase and is subject to potential substantial changes.__

If you experience any issues using MoSeq or have comments on how you'd like to see
MoSeq improved, please [file an issue here](https://github.com/dattalab/moseq2-docs/issues) (but first [see if it's been addressed before](https://github.com/dattalab/moseq2-docs/issues?q=is%3Aissue+is%3Aclosed))
and/or fill out [this user survey](https://forms.gle/FbtEN8E382y8jF3p6).

Currently, we offer 3 notebooks that provide tutorials for using MoSeq's pipeline
to extract, preprocess, model, and perform basic analysis on experimental data. We
hope to expand our selection of notebooks based on user feedback and the
project's trajectory.

### The main MoSeq notebook ([link](./Main-MoSeq2-Notebook.ipynb))

A detailed and complete walk-through of the MoSeq pipeline — from data extraction to modeling.
It includes examples of what to expect at each step, and what to do if things don't go as expected.
There are also a few visualizations to help with extraction and modeling. 

### A notebook to explore modeling and experimental results interactively ([link](./Interactive-Model-Results-Exploration.ipynb))

A stand-alone notebook containing all the interactive data exploration tools for
modeled data. Simply enter the paths to your selected model(s) and index file
to use to syllable exploration tools.

### Flip Classifier Training Notebook ([link](./Flip%20Classifier%20Training%20Notebook.ipynb))

This notebook is a stand-alone tool that can be used to train a new 
flip classifier used to correct the orientation of the extracted animal. If's
utilized if the current flip classifier is not working as expected.
For example, the current flip classifier does not work well with datasets
acquired with the Azure or RealSense depth cameras.

### Our goal for the notebooks

These notebooks are built with the intention that users will copy and adapt them for
their specific needs once they feel comfortable with the MoSeq platform.

## Installation and getting started

### Shortcuts

- [Software requirements](#software-requirements)
- [Installation](#installation)
- [Download test dataset](#downloading-a-test-dataset)
- [Getting started](#getting-started)

### Software Requirements

MoSeq2 is compatible with Windows, MacOS and many flavors of Linux. We've tested
Debian, CentOS, and Ubuntu. Use other Linux distributions at your own risk.

For Windows users, we recommend using Windows Subsystem for Linux (WSL) with the
Ubuntu distribution. This will allow you to use Linux within the Windows
operating system. To install WSL, follow the steps below.
 1. Enable WSL in Windows 10. Official instructions [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and unofficial instructions [here](https://winaero.com/blog/enable-wsl-windows-10-fall-creators-update/)
 2.  Download and install the Ubuntu distribution from the [Microsoft Store](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6?activetab=pivot:overviewtab). This step is included in the official instructions.

The following software packages are required to install and run the MoSeq pipeline.
Don't worry, we'll walk through installation steps for each of these packages in the
[Installation](#installation) section.
 - All Platforms:
     - anaconda3/miniconda3
     - git
     - curl (it should be installed by default on all systems)
     - gcc and g++: accepted and tested versions: 6, 7, and 9
     - [`moseq2-extract==0.6.0`](https://github.com/dattalab/moseq2-extract/blob/release/Documentation.pdf)
     - [`moseq2-pca==0.3.0`](https://github.com/dattalab/moseq2-pca/blob/release/Documentation.pdf)
     - [`moseq2-model==0.4.0`](https://github.com/dattalab/moseq2-model/blob/release/Documentation.pdf)
     - [`moseq2-viz==0.4.0`](https://github.com/dattalab/moseq2-viz/blob/release/Documentation.pdf)
 - macOS:
     - XCode command line tools


## Installation

You're going to need an open terminal to work through this installation guide. On
Mac, use `Terminal.app` or your favorite application. On Windows, open up a terminal
in WSL by starting the distribution application. On Linux, open up the `Terminal`
application.

### Anaconda/miniconda

First, check to see if you have Anaconda already installed in the terminal. Type `conda info` then hit enter. If you received something similar to: `command not found` instead
of a description of the current anaconda installation, you don't have anaconda
installed. Miniconda is the preferred version to install because it only installs
the essentials.

On Linux and WSL:
```bash
curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda3_latest.sh"
chmod +x $HOME/miniconda3_latest.sh  # turn script into excutable
$HOME/miniconda3_latest.sh
```
On MacOS:
```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o "$HOME/miniconda3_latest.sh"
chmod +x $HOME/miniconda3_latest.sh  # turn script into excutable
$HOME/miniconda3_latest.sh
```

We recommend that you say "yes" when the installation prompt asks if you want
to initialize Miniconda3 by running conda init.

You will need to restart your terminal and/or run `conda init` to initialize conda and
have it become recognized as a command. You now installed a new python distribution as
well as some essential tools, like `pip`, which you can use to install other python packages.

If you prefer to install the full Anaconda package, we refer you to the
[official installation documentation](https://docs.anaconda.com/anaconda/install/).

Learn more about `conda` in general [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### gcc

We're going to walk through the process of installing `gcc-7` but you can install any of the
supported versions listed above. Just replace `7` with the version you want to install.

To check if you have `gcc-7`/`g++-7` is installed, run this command:
```bash
which gcc-7 # if you're on macOS
which gcc   # if you're on linux
```
You should expect to see a path to your gcc-7 installation, like this:
```bash
/usr/local/bin/gcc-7  # or /usr/bin/gcc
```

If gcc cannot be found, then follow these steps to install them for your respective OS:

#### For macOS:

1. Install [brew](https://brew.sh/), the macOS package manager, via their website.
2. Install the xcode command line tools. Command: `xcode-select --install`
3. Install gcc: `brew install gcc@7`

Confirm you have `gcc-7` installed by running:
```bash
which gcc-7
```

The following step is important installing the `moseq2-model` dependency.
Once you have confirmed gcc-7 is installed, run the next 2 commands to set
them as default gcc versions. You need to run these two commands for
the installation process only. Not every time you open a new terminal.

```bash
export CC="$(which gcc-7)"
export CXX="$(which g++-7)"
```

#### For WSL/Linux:

How to install on Ubuntu or Debian:
```bash
sudo apt update
sudo apt install build-essential
```
If you're using a different linux distribution, refer to their package manager to
install gcc.

The gcc version that's installed through `build-essential` should be able to compile
`moseq2-model`'s dependencies, and there are no extra steps you need to take to make
MoSeq recognize gcc (like you do for [macOS](#for-macos)).

Alternatively, you can also install `gcc-7.3` with `conda`:
```bash
# automatically sets CC/CXX after installing, too
conda install -c anaconda gcc_linux-64
conda install -c anaconda gxx_linux-64
```

### Git

To check if you have it installed, run `which git`. If it prints a path to git,
you have it installed. Otherwise, refer to the [official installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git/).

:bulb: **Helpful tip!** :bulb:

If you don't want to repeatedly type your GitHub username and password when installing
the MoSeq packages, you can set up `git` to temporarily remember them for 15 minutes:
```bash
git config credential.helper cache
```

### MoSeq2 packages and notebook

Now that we've handled the prerequisites, we can finally install MoSeq.

The optimal way to install MoSeq is automagically via a script we supply. But if you
run into trouble, you can run each of the commands from the script separately
to figure out where the problem lies. If you run into installation issues that you
can't figure out on your own or aren't documented here, please [sumbit an issue](https://github.com/dattalab/moseq2-docs/issues)

#### Pre-installation instructions

- download the `moseq2-app` GitHub repository, and navigate to it:
```bash
git clone -b jupyter https://github.com/dattalab/moseq2-app.git
cd moseq2-app
```
You'll be asked for your github username and password since this is a private repository.

- We recommend that you create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) named `moseq2-app`:
```bash
# assuming you're in the moseq2-app directory:
conda env create -n moseq2-app --file scripts/moseq2-env.yaml
```
If you're already a conda master, feel free to install MoSeq in any environment you want.
Just make sure the dependencies described in `scripts/moseq2-env.yaml` are installed in
that environment.

- activate the new conda environment
```bash
conda activate moseq2-app
```

- If on a mac, link to the gcc compilers via 2 environment variables, like [we said
above](#for-macos) (remember to replace `7` with the gcc version you have)
```bash
export CC="$(which gcc-7)"
export CXX="$(which g++-7)"
```

#### Installing MoSeq

Make sure you're in the `moseq2-app` folder with your new conda environment activated
([see pre-installation instructions](#pre-installation-instructions))

Run:
```bash
./scripts/install_moseq2_app.sh
```

If you get any errors running that script, open the script in a text file and run each line
of the script independently. This will help you (and us) figure out where the
error is occurring.

## Downloading a test dataset

To try MoSeq2 on some sample data, we have provided 2 scripts to either download
the complete 48 session dataset, or a smaller dataset with 20 total sessions,
10 of each experimental group.

If on WSL/linux (Ubuntu/Debian), you might need to install the `unzip` package:
```bash
sudo apt install unzip
```

Run the following command in the `moseq2-app` directory
```bash
./scripts/download_small_dataset.sh  # or ./scripts/download_full_dataset.sh
```

The shell scripts will create a new folder within `moseq2-app` to download the data.
Be warned, this process will take a long time to finish (hours), and requires 20-60 GB
of disk space.

## Getting started

At this point, you should have MoSeq installed and some data to extract.
As discussed above, we've provided 3 jupyter notebooks that describe
MoSeq's pipeline in detail. To use the notebooks, make sure that you've
activated the conda environment you installed MoSeq in (likely `moseq2-app`)
and you have navigated to the folder that contains the jupyter notebook(s).
__:exclamation: IMPORTANT: :exclamation: Make sure that the notebook is run from the same directory as 
your dataset so that the videos you generate will load into the notebooks
properly.__ 

Run the following command to launch the jupyter notebook:
```bash
jupyter notebook
```

Then, when your browser launches, click on the notebook you want to run.
The notebooks contain tutorial materials to walk you through the process
of running the MoSeq pipeline itself. Our hope is that once you've grown
accustomed to the pipeline, you will be able to modify (i.e., remove
the text and examples) the notebook to increase your productivity and
to fit your needs best. Please share with us how you've adapted the MoSeq
pipeline for your research!

## (Optional) Building and re-generating documentation

All documentation regarding moseq2-extract can be found in the `Documentation.pdf` file in the root directory,
an HTML ReadTheDocs page can be generated via running the `make html` in the `docs/` directory.

MoSeq2 uses `sphinx` to generate the html and pdf documentation. To install `sphinx`, follow the commands below:
```.bash
pip install sphinx==3.0.3
pip install sphinx-rtd-theme
pip install rst2pdf
``` 

To generate a PDF version of the documentation, simply run `make pdf` in the `docs/` directory.

For information on getting started, check out the [MoSeq Wiki](https://github.com/dattalab/moseq2-app/wiki).

## Bug Reporting

If you experience any errors during installation, consult the [troubleshooting](./TROUBLESHOOT.md) guide.
If the your issue is not resolved there, submit a GitHub issue.

To report any issues or bugs using the notebook(s), please refer to the GitHub issues page in this repository:
[Report Issue](https://github.com/dattalab/moseq2-app/issues/new).
