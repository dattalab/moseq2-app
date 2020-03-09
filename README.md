# MoSeq2-Notebook: An interactive Jupyter Notebook for animal behavior sequencing

This package contains a jupyter notebook that is tailored for novice programmers to process
their depth videos of rodents, and segment their behavior into what is denoted as "syllables".

The notebook is a detailed walk-through the entire analysis pipeline.
It includes examples of what to expect at each step, and what to do if things don't go as expected. 

The Moseq2 toolkit enables users to model rodent behavior across different experimental groups, and
measure the differences between their behavior usages, durations, transition patterns. etc.

***

# MoSeq2 Software Requirements

In order to successfully install MoSeq2, we recommend you ensure you have the following libraries and packages already installed:
 - All Platforms:
     - anaconda3/miniconda3
     - python3.6 (top right-hand side of the jupyter notebook will indicate the python version for you)
     - git
     - gcc-7 and g++-7 (ensure this version is default)
     - numpy
     - pip
     - Conda Environment:
         - ffmpeg
     - CentOS:
         - libSM
     - MacOS:
         - latest version of XCode

# Installation

To install and set up your MoSeq environment using `easy_install.sh`, ensure you have both anaconda3/miniconda3 and gcc-7/g++-7 installed.

Once both are installed, simply execute the easy install shell file: `./easy_install.sh`

The install script will also automatically try to ensure GCC is being installed for the 
following operating systems: Darwin (Mac), Linux, Cygwin (Windows).

# Get Started

After the easy_install is complete, activate your new conda environment and launch the jupyter notebook to get started.

```bash
conda activate moseq2dev

jupyter notebook
```  