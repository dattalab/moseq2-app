[![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XPHHifxcxChBFeDGgSmjllFdpQaZ4z9d?usp=sharing)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Wiki-blue)](https://github.com/dattalab/moseq2-app/wiki)

Last Updated: 08/19/2021

# Overview
<center><img src="https://drive.google.com/uc?export=view&id=1kHdmkBx_XlueTJocREDx4YeHGrjfYKJv"></center>
The MoSeq2 toolkit enables users to model mouse behavior across different experimental groups, and
measure the differences between their behavior usages, durations, transition patterns. etc.

This package contains functionalities that can be used interactively in jupyter notebooks. 
We provide a series of Jupyter Notebooks that cover the entire MoSeq pipeline for novice 
programmers to process their depth videos of mice, and segment their behavior into what is denoted as "syllables".

<!---
All colab notebooks right now are temporary shared from my Google drive. 
Users will get the "click one link and everything works" experience once the notebooks are public on Github. 
--> 
Consult the wiki-page for more detailed documentation of the MoSeq pipeline [here](https://github.com/dattalab/moseq2-app/wiki).
You can try MoSeq on [Google Colab](https://colab.research.google.com/drive/1XPHHifxcxChBFeDGgSmjllFdpQaZ4z9d?usp=sharing) on our test data or your data on Google Drive.

# [Documentation: MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki)
You can find more information about MoSeq Pipeline, step-by-step instructions, documentation for Command Line Interface(CLI), tutorials etc in [MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki).

# Installation
Installing MoSeq2 and the dependencies requires **Python version 3.7, gcc-7 and g++-7**. We recommand using our conda environment yaml file to install MoSeq. Run the following to clone the repository and install MoSeq2 and the Jupyter extensions:
<!---Current branch is dev but will update that to release later.-->
```bash
git clone -b dev https://github.com/dattalab/moseq2-app.git
cd moseq2-app
conda create -n moseq2-app --file scripts/moseq2-env.yaml
./scripts/install_moseq2_app.sh
```
### We provide step-by-step guides for installing MoSeq in a conda environment or in Docker in the wiki [here](https://github.com/dattalab/moseq2-app/wiki/MoSeq2-Installation).

# Test data and Colab Demo
You can try the entire MoSeq pipeline on [Google Colab](https://colab.research.google.com/drive/1XPHHifxcxChBFeDGgSmjllFdpQaZ4z9d?usp=sharing) using the test data. 
If you want to try the test data locally, you can find the test data and their desciptions [here](https://github.com/dattalab/moseq2-app/wiki/Download-Test-Data). 
<!---Maybe put some crowd movies here?-->

# Community Support and Contributing
bug report, feature report, pr, and slack and guidline.
- Please join [![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com) to post qustions and interactive with MoSeq Developers and users.
- If you encounter bugs, errors or issues, please submit a Bug report [here](https://github.com/dattalab/moseq2-app/issues/new/choose). We encourage you to check out the [troubleshooting and tips section](https://github.com/dattalab/moseq2-app/wiki/Troubleshooting-and-Tips) and search your issues in [the existing issues](https://github.com/dattalab/moseq2-app/issues) first.   
- If you want to see certain features in MoSeq or you have new ideas, please submit a Feature request [here](https://github.com/dattalab/moseq2-app/issues/new/choose).
- If you want to contribute to our codebases, please checkout our [Developer Guidelines](https://github.com/dattalab/moseq2-app/wiki/MoSeq-Developer-Guidelines).
- Please tell us what you think by fiiling out [this user survey](https://forms.gle/FbtEN8E382y8jF3p6).

# Versions
<!---The current changelog is really messy, we either remove it entirely or clean it up. We should remove the road map too-->
- Current verison: Version 0.2.1
- [Changelog](https://github.com/dattalab/moseq2-app/wiki/Changelog)

# License
<!---We should probably pick a license-->

# Events & News
<!---Future events, project related news etc-->
- Lorem ipsum dolor sit amet
- consectetur adipiscing elit
- Suspendisse ornare faucibus vestibulum
- Etiam vel elit ac lorem bibendum suscipit

# Publications
<!---I think we have more than just these-->
- [Mapping Sub-Second Structure in Mouse Behavior](http://datta.hms.harvard.edu/wp-content/uploads/2018/01/pub_23.pdf)
- [The Striatum Organizes 3D Behavior via Moment-to-Moment Action Selection](http://datta.hms.harvard.edu/wp-content/uploads/2019/06/Markowitz.final_.pdf)
- [Revealing the structure of pharmacobehavioral space through motion sequencing](https://www.nature.com/articles/s41593-020-00706-3)
- [Q&A: Understanding the composition of behavior](http://datta.hms.harvard.edu/wp-content/uploads/2019/06/Datta-QA.pdf)
