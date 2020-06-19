# MoSeq2-Notebook: An interactive Jupyter Notebook for animal behavior sequencing

Last Updated: 06/10/2020

Consult the wiki-page for a complete overview and description of the MoSeq pipeline [here](https://github.com/dattalab/moseq2-app/wiki).

This package contains a jupyter notebook that is tailored for novice programmers to process
their depth videos of rodents, and segment their behavior into what is denoted as "syllables".

The notebook is a detailed walk-through the entire analysis pipeline.
It includes examples of what to expect at each step, and what to do if things don't go as expected. 

The MoSeq2 toolkit enables users to model rodent behavior across different experimental groups, and
measure the differences between their behavior usages, durations, transition patterns. etc.

__Note: The Notebook is currently in a testing phase and is utilizing the `test-suite` branch from all of the MoSeq2
utility repositories.__

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
     - python3.6 (top right-hand side of the jupyter notebook will indicate the python version for you)
     - git
     - gcc-7 and g++-7 (ensure this version is installed and defaulted)
     - numpy
     - pip
     - Conda Environment:
         - ffmpeg
     - CentOS:
         - libSM
     - MacOS:
         - latest version of XCode

Below is a list of all the required minimum versions of each repository to ensure are installed:
 - `moseq2-extract==0.5.0`
 - `moseq2-pca==0.3.0`
 - `moseq2-model==0.4.0`
 - `moseq2-viz==0.3.0`

# Installation

### Ensure Anaconda for Linux is Installed
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


### Ensure the Correct Version of GCC-7/G++-7 is defaulted
__Note: gcc version `6.2.0` is also acceptable.__

To check if you have gcc-7/g++-7 is installed, run this command: `which gcc-7`.
You should expect to see an outputted path to your gcc-7 installation, like this: `/usr/local/bin/gcc-7`

If gcc-7/g++-7 cannot be found, then follow these steps to install them for your respective OS:

For MacOS:
 - You can use [brew](https://brew.sh/) to install gcc by running these commands:
```bash
xcode-select --install # Download the latest version of Xcode if you don't already have it
brew install gcc@7
```

For WSL/Ubuntu/Linux, run the following commands:
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
 - Once you have gcc-7 installed and is confirmed by running `which gcc-7`/`which g++-7`, run the next 2 commands
to set them as default gcc versions.

```bash
export CC="$(which gcc-7)"
export CXX="$(which g++-7)"
```

### Create MoSeq2 Conda Environment and Install Dependencies  


Once conda is operational and gcc-7 is installed, set up and install your MoSeq environment by running the following commands:
```bash
git clone -b release https://github.com/dattalab/moseq2-app.git
cd moseq2-app
./easy_install.sh
conda activate moseq2-app
pip install git+https://github.com/dattalab/moseq2-model.git@test-suite
```
***

# Download a Test Dataset

To try MoSeq2 on some sample data, we have provided 2 bash script files to either download the complete 48 session dataset,
or 20 total sessions with 10 of each experimental group. Download your chosen dataset using either of the following command:
```bash
./download_10n10.sh # 20 total session; 10 saline, 10 amphetamine

./download_full_dataset.sh # all 48 sessions [24 sessions per group]
```
The shell scripts will create a new directory in this cloned repo with a copy of the `MoSeq2-Notebook`. 
Once, the download is complete, navigate to that directory and launch the jupyter notebook (within the newly created conda env). 

__Note: while using this dataset, you may use all the default parameter settings in the Notebook as they were previously 
configured to match this dataset. I.e. you can click 'Run all cells' in the beginning and wait for the analysis pipeline to complete.__
 
# Get Started

Once everything is installed, it is recommended that you copy the jupyter notebook to the directory containing 
your data to analyze and run the notebook from there, such that you preserve the original copy in your cloned directory.

__Note: ensure that the `jupyter notebook` command is being run from the same directory as the notebook itself.__

```bash
conda activate moseq2-app

jupyter notebook
```  
