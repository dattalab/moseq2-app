# MoSeq2-Notebook: An interactive Jupyter Notebook for animal behavior sequencing

<center><img src="https://drive.google.com/uc?export=view&id=1jCpb4AQKBasne2SoXw_S1kwU7R9esAF6"></center>

Last Updated: 10/06/2020

Consult the wiki-page for a complete overview and description of the MoSeq pipeline [here](https://github.com/dattalab/moseq2-app/wiki).

This package contains interactive jupyter notebooks that are tailored for novice programmers to process
their depth videos of rodents, and segment their behavior into what is denoted as "syllables".

Below is a list of the notebooks afforded by MoSeq2-App:
- The Main MoSeq2-Notebook: A full detailed walk-through of the entire MoSeq pipeline with interactive tools to
to handle and validate the data extraction process, and interactively explore the modeled behavior results.
  - It includes examples of what to expect at each step, and what to do if things don't go as expected.
- The Interactive-Model-Results Explorer Notebook: A stand-alone notebook containing all the interactive 
data exploration tools for modeled data.
  - Simply enter the paths to your selected model(s) and index file to use to syllable exploration tools.
- Flip Classifier Training Notebook: This notebook is a stand-alone tool that is optionally used to train new 
flip classifiers (used to accurately extract mouse orientation) for currently unsupported data acquisition setups. 
For example, Azure or RealSense captured mice/rats with or without head fixed cables.

The MoSeq2 toolkit enables users to model rodent behavior across different experimental groups, and
measure the differences between their behavior usages, durations, transition patterns. etc.

__Note: The Notebook is currently in a testing phase and is utilizing the `release` branch from all of the MoSeq2
utility repositories.__

***

## Shortcuts
- [Software Requirements](#moseq2-software-requirements)
- [Install Dependencies](#installation)
- [Install MoSeq2-App](#install-moseq2-app-env-or-package)
- [Download Test Dataset](#download-a-test-dataset)
- [Get Started](#get-started)

***

# MoSeq2 Software Requirements

MoSeq2 is compatible with Windows, MacOS and Linux.

For Windows users, we recommend using an Ubuntu shell in the Windows Subsystem for Linux.
For information on how to download and setup Ubuntu on Windows, follow these steps:
 1. [Enable WSL in Windows 10](https://winaero.com/blog/enable-wsl-windows-10-fall-creators-update/)
 2.  Download and install Ubuntu from the [Microsoft Store](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6?activetab=pivot:overviewtab).

MoSeq2 requires the following platform dependencies to be installed:
 - All Platforms:
     - anaconda3/miniconda3 __(FOR LINUX)__
     - python>=3.6;<3.8 (top right-hand side of the jupyter notebook will indicate the python version for you)
     - git
     - wget
     - gcc-7 and g++-7 (other accepted versions: [6.2.0, 7.5.0, gcc-9/g++-9])
     - numpy
     - pip
     - Conda Environment:
         - ffmpeg
 - CentOS:
     - libSM
 - MacOS:
     - latest version of XCode

Below is a list of all the required minimum versions of each repository to ensure are installed:
 Below is a list of all the required minimum versions of each repository to ensure are installed:
 - [`moseq2-extract==0.6.0`](https://github.com/dattalab/moseq2-extract/blob/release/Documentation.pdf)
 - [`moseq2-pca==0.3.0`](https://github.com/dattalab/moseq2-pca/blob/release/Documentation.pdf)
 - [`moseq2-model==0.4.0`](https://github.com/dattalab/moseq2-model/blob/release/Documentation.pdf)
 - [`moseq2-viz==0.4.0`](https://github.com/dattalab/moseq2-viz/blob/release/Documentation.pdf)

***

# Installation

If you already have Anaconda installed, ensure you are working in a separate environment so
 that your pre-existing work flows are not disrupted by any of the newly set environmental variables
 or jupyter settings.
 
## Anaconda Installation Guide
### Ensure Anaconda For Linux is Installed
The reason it is recommended to use Mini/Anaconda for Linux instead of installing a separate Conda application is due
 to the need to set environmental variables relating to respective OS dependencies. Using Linux provides more control
 and flexibility when dealing with OS-specific dependencies across different Operating Systems. 

To check if you have anaconda/miniconda already installed, you can run the following command in your
respective CLI: `conda info`. 

If the command doesn't return anything, then continue with the install steps below:
 - The first command will download the miniconda install script `miniconda3_latest.sh` to your default landing directory.
```bash
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda3_latest.sh"
chmod +x $HOME/miniconda3_latest.sh
$HOME/miniconda3_latest.sh -b -p $HOME/miniconda3
```

OR

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x $HOME/Anaconda3-2020.02-Linux-x86_64.sh
$HOME/Anaconda3-2020.02-Linux-x86_64.sh -b -p $HOME/anaconda3
```

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
If so, activate the environment and run the install script again but enter `2` to only install the latest dependency versions.

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

## Bug Reporting
To report any issues or bugs using the notebook(s), please refer to the GitHub issues page in this repository:
[Report Issue](https://github.com/dattalab/moseq2-app/issues/new).