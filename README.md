# MoSeq2-Notebook: A suite of interactive Jupyter notebooks for animal behavior sequencing and analysis

[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/dattalab/community)

<!-- <center><img src="https://drive.google.com/uc?export=view&id=1jCpb4AQKBasne2SoXw_S1kwU7R9esAF6"></center> -->

Last Updated: 10/06/2020

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

### Flip Classifier Training Notebook

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
- [Download test dataset](#download-a-test-dataset)
- [Getting started](#get-started)

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
     - anaconda3/miniconda3 __(FOR LINUX)__
     - python>=3.6,<3.8
     - git
     - wget or curl
     - gcc and g++: accepted and tested versions: 6, 7, and 9
     - numpy
     - pip
     - Conda Environment:
         - ffmpeg==4.2.0
     - [`moseq2-extract==0.6.0`](https://github.com/dattalab/moseq2-extract/blob/release/Documentation.pdf)
     - [`moseq2-pca==0.3.0`](https://github.com/dattalab/moseq2-pca/blob/release/Documentation.pdf)
     - [`moseq2-model==0.4.0`](https://github.com/dattalab/moseq2-model/blob/release/Documentation.pdf)
     - [`moseq2-viz==0.4.0`](https://github.com/dattalab/moseq2-viz/blob/release/Documentation.pdf)
 - CentOS:
     - libSM
 <!-- - MacOS:
     - latest version of XCode -->


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
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda3_latest.sh"
chmod +x $HOME/miniconda3_latest.sh  # turn script into excutable
$HOME/miniconda3_latest.sh -b -p $HOME/miniconda3
```
On MacOS:
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o "$HOME/miniconda3_latest.sh"
chmod +x $HOME/miniconda3_latest.sh  # turn script into excutable
$HOME/miniconda3_latest.sh -b -p $HOME/miniconda3
```
**NOTE: the `-b` flag installs miniconda in silent mode and assumes you accept
their license agreement. For more information, refer to the
[official documentation](https://docs.anaconda.com/anaconda/install/silent-mode/)**

If you prefer to install the Anaconda package, we refer you to the [official installation documentation](https://docs.anaconda.com/anaconda/install/).

### Install GCC Dependency
__Remember, if you are going to use an alternate version of gcc, change inputted the commands to match your version.__

Below are README shortcuts to install GCC for your respective operating system:
- [For MacOS](#For-MacOS)
- [For WSL/Ubuntu/Linux](#for-wslubuntulinux)
- [Set gcc version as default](#set-the-gcc-version-as-default)
 
To check if you have gcc-7/g++-7 is installed, run this command:
```bash
which gcc-7
# or `which gcc` if your default version is in the working set gcc versions. 
```
You should expect to see an outputted path to your gcc-7 installation, like this:
```bash
/usr/local/bin/gcc-7
```

If gcc-7/g++-7 cannot be found, then follow these steps to install them for your respective OS:

#### For MacOS:
 - You can use [brew](https://brew.sh/) to install gcc by running these commands:
 
To install Homebrew:
```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

GCC Install Command:
```bash
xcode-select --install || softwareupdate -i -a # Download the latest version of Xcode if you don't already have it
brew install gcc@7
```

#### For WSL/Ubuntu/Linux:
```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-7 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7 
sudo update-alternatives --config gcc
gcc --version
g++ --version
```

Finally, for all platforms:

Confirm you have `gcc-7` installed by running:
```bash
which gcc-7
which g++-7
```

The following step is important installing the `moseq2-model` dependency.

### Set the GCC Version as Default
Once you have confirmed gcc-7 is installed, run the next 2 commands to set them as default gcc versions.
It is also recommended to copy these two commands to your `~/.bashrc` file (for Linux and Windows)  
or `~/.bash_profile` file (for MacOS) to preserve this setting from now on. 

__NOTE: If you are not using `gcc-7` remember to alter the command below to reflect your version.__

```bash
export CC="$(which gcc-7)"
export CXX="$(which g++-7)"
```

## Install MoSeq2-App Env or Package
Once conda is operational and gcc-7 is installed, clone this repository and run the install script (line 3 below).
The install script gives the option to either create a new conda environment for moseq2-app, or to install the latest versions
of the required dependencies into a pre-existing __(activated)__ conda environment.
```bash
git clone -b release https://github.com/dattalab/moseq2-app.git
cd moseq2-app
./scripts/easy_install.sh # '1' to create a new env, '2' to install the latest version of moseq2-app in current env
# if a new env was created
source ~/.bashrc # or ~/.bash_profile  [OPTIONAL: depending on whether env can be found after creation]
conda activate moseq2-app 
./scripts/install_moseq2_app.sh
```

If you created a new environment, you may need to restart your shell in order for it to become visible. 
This is done in the 5th line in the command block above `source ~/.bashrc`. 

If for whatever reason the environment creation is interrupted, restart the shell and check if the environment `moseq2-app` exists.
If so, activate the environment and run the `easy_install.sh` script again but enter `2` to only install the latest dependency versions.

You can list your currently existing conda environments by running the following command:
```bash
conda env list
```

***

# Download a Test Dataset

To try MoSeq2 on some sample data, we have provided 2 bash script files to either download the complete 48 session dataset,
or 20 total sessions with 10 of each experimental group.

`wget` is required in order to download the datasets. Installation commands for different operating systems are listed below.

For MacOS:
```bash
brew install wget
```

For Ubuntu/Debian:
```bash
sudo apt-get install wget
```

Download your chosen dataset using either of the following command (might take >30 minutes):
```bash
./download_10n10.sh # 20 total session; 10 saline, 10 amphetamine

./download_full_dataset.sh # all 48 sessions [24 sessions per group]
```

The shell scripts will create a new directory in this cloned repo with a copy of the `Main-MoSeq2-Notebook`. 
Once, the download is complete, navigate to that directory and launch the jupyter notebook (within the activated conda env).

***

## Get Started
To use the notebooks, make sure your environment is activated and your current directory includes 
the notebooks you would like to use. 

__NOTE: It is important for the notebook to be run from the parent directory 
of your dataset in order for the videos to be displayed.__ 

Run the following command to launch the jupyter notebook:
```bash
jupyter notebook
```  

## Documentation

MoSeq2 uses `sphinx` to generate the documentation in HTML and PDF forms. To install `sphinx`, follow the commands below:
```.bash
pip install sphinx==3.0.3
pip install sphinx-rtd-theme
pip install rst2pdf
``` 

All documentation regarding moseq2-extract can be found in the `Documentation.pdf` file in the root directory,
an HTML ReadTheDocs page can be generated via running the `make html` in the `docs/` directory.

To generate a PDF version of the documentation, simply run `make pdf` in the `docs/` directory.

For information on getting started, check out the [MoSeq Roadmap](https://github.com/dattalab/moseq2-docs/wiki).


## Bug Reporting
If you experience any errors during installation, consult the [`TROUBLESHOOT.md`](https://github.com/dattalab/moseq2-app/blob/flip-training-notebook/TROUBLESHOOT.md) file.
If the your issue is not resolved there, submit a GitHub issue.

To report any issues or bugs using the notebook(s), please refer to the GitHub issues page in this repository:
[Report Issue](https://github.com/dattalab/moseq2-app/issues/new).
