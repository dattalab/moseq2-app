# MoSeq2-Notebook Setup
MoSeq2 Web Application Platform used to run all of the MoSeq2 tools in a Jupyter Notebook.

**Follow the installation and set up steps outlined in this README to ensure your jupyter notebooks will run smoothly.**

__NOTE: If you already have anaconda installed AND have an environment setup: simply install the latest dev branches in your activated environment, and run the commands below sequentially.__ 

```bash
conda activate moseq2dev # moseq2dev being a pre-existing conda environment 

export CC=/usr/local/bin/gcc-7
export CXX=/usr/local/bin/g++-7

conda install -c conda-forge ffmpeg

pip install --upgrade pip

pip install git+https://github.com/dattalab/moseq2-extract.git@dev

pip install git+https://github.com/dattalab/moseq2-pca.git@dev

pip install git+https://github.com/dattalab/moseq2-model.git@dev

pip install git+https://github.com/dattalab/moseq2-viz.git@dev
```

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
         
## Installing and Setting Up Miniconda3

**If you already have anaconda/miniconda installed, you can skip this step.**

### Installation
**On Linux, copy the following commands into your terminal:**
```bash

curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda3_latest.sh"

chmod +x $HOME/miniconda3_latest.sh

$HOME/miniconda3_latest.sh -b -p $HOME/miniconda3

```

**On MacOS, copy the following commands into your terminal:**
```bash

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o "$HOME/miniconda3_latest.sh"

chmod +x $HOME/miniconda3_latest.sh

$HOME/miniconda3_latest.sh -b -p $HOME/miniconda3

```

**On Windows 10, copy the following commands into your terminal:**

Ensure you have the [Debian WSL](https://www.microsoft.com/en-us/p/debian/9msvkqc78pk6?activetab=pivot%3Aoverviewtab) installed from the Windows store which will allow you to run a bash terminal. 

Note: You may have to manually enable Windows Subsystem for Linux (WSL) before being able to use the bash terminal. [Follow this guide to setup your WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

Once it is set up, copy the following commands into your terminal:
```bash

sudo apt-get update

sudo apt-get install git build-essential curl libxrender-dev libsm6 libglib2.0-0

curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda3_latest.sh"

chmod +x $HOME/miniconda3_latest.sh

$HOME/miniconda3_latest.sh -b -p $HOME/miniconda3

```

 ## Miniconda Setup (Optional)
 
 Add the appropriate lines to your __.bashrc__ file (or __.bash_profile__ on MacOS)

For windows users:
```bash
cat >> ~/.bashrc << END
PATH=\$HOME/miniconda3/bin:\$PATH
END
source $HOME/.bashrc
```

For macOS users:
```bash
cat >> ~/.bash_profile << END
PATH=\$HOME/miniconda3/bin:\$PATH
END
source $HOME/.bash_profile
```

__NOTE FOR NOVICE TERMINAL USERS: you must run the last line of the above code block when launching a new terminal session. We recommend adding the command to your .bashrc file so that the 'conda' command is always available.__
The reason for this is because .bashrc is executed for non-login profiles while .bash_profile is only executed for login profiles. This is optional and for your convenience.

Example of how to do that below (for MacOS/.bash_profile users):
```bash
sudo chmod 770 .bashrc ##IF you do not have user-level permission to edit the file already
cat >> ~/.bashrc << END
source $HOME/.bash_profile
END
```

### Create Conda Virtual Environment
In order to create and start using your miniconda environment, run the following commands:
```bash
conda create -n "moseq2dev" python=3.6 -y
conda activate moseq2dev
```

#### Ensure you are running the python version located in your corresponding conda environment.

Run the following command in your terminal:
```bash
which python
```

Before continuing, this is what your output should look like: ```/Users/username/anaconda3/envs/moseq2dev/bin/python```

If that is not the case, then close this notebook session, activate the `moseq2dev` conda environment in the terminal.


#### Install conda dependency

Make sure you install ffmpeg in the moseq2 environment, since the pipeline makes heavy use of it.
Run the following bash install command:
```bash
conda install -n moseq2dev -c conda-forge ffmpeg
```

Now that your conda environment is set up, lets ensure your C/C++ compiler versions are correct in order to install all the tools successfully.

### Setting a default gcc/g++ compiler (Very Important)

MoSeq2 requires using a legacy version of gcc/g++: version 7 or 8.

To check which version of gcc is currently used run:
```bash
gcc -v
```

If your gcc version is the latest version, it may cause problems while running MoSeq, so we recommend
following these steps:

__MAC USERS NOTE: if you do not have the latest version of xcode, you will not be able to install your preferred compiler. To do this, run the following command.__
```bash
xcode-select --install
brew install gcc@7
```

**Note: this step is important for pyhsmm (in moseq2-model) to properly install and run in your virtual environment.**

To set your compilers to the correct version run: 
```bash
export CC=/usr/local/bin/gcc-7
export CXX=/usr/local/bin/g++-7
```

# Installing latest MoSeq2 Repositories

Once you have set up your environment and installed all the prerequisite packages, you are now ready to install the MoSeq2 tools.
```bash
source activate moseq2dev

pip install --upgrade pip

pip install git+https://github.com/dattalab/moseq2-extract.git@dev

pip install git+https://github.com/dattalab/moseq2-pca.git@dev

pip install git+https://github.com/dattalab/moseq2-model.git@dev

pip install git+https://github.com/dattalab/moseq2-viz.git@dev
```

# Begin Analysis
Once successfully completed installation, you are now ready to use MoSeq2! 

To get started,
- Start your specified conda environment (if you haven't already).
- Run the Jupyter Notebook.