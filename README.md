[![Build Status](https://app.travis-ci.com/dattalab/moseq2-app.svg?token=ptXpSa3Fp9PKqkxJDkDr&branch=dev)](https://app.travis-ci.com/dattalab/moseq2-app)
[![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XPHHifxcxChBFeDGgSmjllFdpQaZ4z9d?usp=sharing)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Wiki-blue)](https://github.com/dattalab/moseq2-app/wiki)

Last Updated: 08/19/2021

# Overview

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1oc0_0mlN0VTZEPMQTg_hnYAC87Lb58MI" />
</p>

<!---Adapted from 
Datta, Sandeep Robert, David J. Anderson, Kristin Branson, Pietro Perona, and Andrew Leifer. 2019. “Computational Neuroethology: A Call to Action.” Neuron 104 (1): 11–24.
If this looks good, I will add citation links to the mentioned paper
-->
Motion Sequencing (MoSeq) is an unsupervised machine learning method to describe behavior. MoSeq takes 3D depth videos as input and then uses statistical learning techniques to fit an autoregressive hidden Markov model that parses behavior into a set of sub-second motifs called syllables. By combining MoSeq with electrophysiology, multi-color photometry, and miniscope methods, neural correlates for 3D behavioral syllables have recently been identified in dorsolateral striatum (DLS) ([Markowitz et al., 2018](https://www.sciencedirect.com/science/article/pii/S0092867418305129)). Furthermore, MoSeq has been combined with optogenetic stimulation to reveal the differential consequences of activating the motor cortex, the dorsal striatum, and the ventral striatum ([Pisanello et al.,2017](https://www.nature.com/articles/nn.4591); [Wiltschko et al., 2015](https://www.sciencedirect.com/science/article/pii/S0896627315010375)). These results are consistent with similar results recently obtained using marker-based approaches to explore the relationship between 3D posture and activity in posterior parietal cortex ([Mimica et al., 2018](https://www.science.org/doi/10.1126/science.aau2013)).

## MoSeq2 Toolkit
<!---Need to check the moseq pipeline image and see if there is any copy right issue. We could/probably should make our own-->
The MoSeq2 toolkit enables users to model mouse behavior across different experimental groups, and
measure the differences between their behavior usages, durations, transition patterns. etc.

This package contains functionalities that can be used interactively in jupyter notebooks. 
We provide a series of Jupyter Notebooks that cover the entire MoSeq pipeline to process their depth videos of mice, and segment their behavior into what is denoted as "syllables". In addition to the Jupyter notebooks, MoSeq has Google Colab notebooks and a Command Line Interface. Consult the wiki page for more detailed documentation of the MoSeq pipeline [here](https://github.com/dattalab/moseq2-app/wiki).
You can try MoSeq on [Google Colab](https://colab.research.google.com/drive/1JOFvvUIfQlhjWZ3MZ3yZ0_hryhI-u55U?usp=sharing) on our test data or your data on Google Drive.
<!---
All colab notebooks right now are temporarily shared from my Google drive. 
Users will get the "click one link and everything works" experience once the notebooks are public on Github. 
--> 
## Which one do I use?
**Jupter Notebook**
|Pros                                                   |Cons                                          |
|-------------------------------------------------------|----------------------------------------------|
|Easy to use                                            |Extract all sessions sequentially (slow)      |
|Shows both the code blocks and the output              |Doesn't support automation and parallelization|
|Comes with interactive widgets to analyze model results|Less tunable parameters than the CLI          |

**Google Colab Notebook**
|Pros                                               |Cons                                                                                            |
|---------------------------------------------------|------------------------------------------------------------------------------------------------|
|No local installation and environment set up needed|Extract all sessions sequentially (slow)                                                        |
|Comes with MoSeq test data to try MoSeq out        |Google Colab has usage limit so the runtime may be disconnected before the pipeline finishes    |
|Could be used directly on data in Google Drive     |All the necessary packages need to be installed every time at the beginning of a runtime session|

**Command Line Interface (CLI)**
|Pros                                                                     |Cons                                                         |
|-------------------------------------------------------------------------|-------------------------------------------------------------|
|Supports extracting sessions parallelly (fast)                           |Could be confusing for users that have never used a CLI      |
|Faster, more efficient and more tunable paramters                        |Limited visualization capabilities and no interactive widgets|
|Could be used in bash scripts flexibly for automation and parallelization|                                                             |

If you are interested in using the CLI for extraction and modeling, but using the interactive widgets in the Jupyter notebooks to find parameters and analyze results interactively, you can find more information in [CLI extraction and modeling documentation](https://github.com/dattalab/moseq2-app/wiki/Command-Line-Interface-for-Extraction-and-Modeling) and [Interactive Model Results Exploration Notebook documentation](https://github.com/dattalab/moseq2-app/wiki/Interactive-Model-Results-Exploration-Notebook-Instructions).

# [Documentation: MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki)
You can find more information about MoSeq Pipeline, installation, step-by-step instructions, documentation for Command Line Interface(CLI), tutorials etc in [MoSeq2 Wiki](https://github.com/dattalab/moseq2-app/wiki).

# Installation
Installing MoSeq2 and the dependencies requires **Python version 3.7, gcc-7 and g++-7**. We recommend using our conda environment yaml file to install MoSeq. Run the following to clone the repository and install MoSeq2 and the Jupyter extensions:

```bash
git clone -b dev https://github.com/dattalab/moseq2-app.git
cd moseq2-app
conda create -n moseq2-app --file scripts/moseq2-env.yaml
./scripts/install_moseq2_app.sh
```
### We provide step-by-step guides for installing MoSeq in a conda environment or Docker in the wiki [here](https://github.com/dattalab/moseq2-app/wiki/MoSeq2-Installation).

# Test data and Colab Demo
You can try the entire MoSeq pipeline on [Google Colab](https://colab.research.google.com/drive/1JOFvvUIfQlhjWZ3MZ3yZ0_hryhI-u55U?usp=sharing) using the test data. 
If you want to try the test data locally, you can find the test data and their descriptions [here](https://github.com/dattalab/moseq2-app/wiki/Download-Test-Data). 


# Community Support and Contributing
- Please join [![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com) to post questions and interactive with MoSeq developers and users.
- If you encounter bugs, errors or issues, please submit a Bug report [here](https://github.com/dattalab/moseq2-app/issues/new/choose). We encourage you to check out the [troubleshooting and tips section](https://github.com/dattalab/moseq2-app/wiki/Troubleshooting-and-Tips) and search your issues in [the existing issues](https://github.com/dattalab/moseq2-app/issues) first.   
- If you want to see certain features in MoSeq or you have new ideas, please submit a Feature request [here](https://github.com/dattalab/moseq2-app/issues/new/choose).
- If you want to contribute to our codebases, please check out our [Developer Guidelines](https://github.com/dattalab/moseq2-app/wiki/MoSeq-Developer-Guidelines).
- Please tell us what you think by filling out [this user survey](https://forms.gle/FbtEN8E382y8jF3p6).

# Versions
- Current version: Version 0.2.1
- [Changelog](https://github.com/dattalab/moseq2-app/wiki/Changelog)

# License
<!---We should probably pick a license-->

# Events & News
<!---Future events, project related news etc-->
Coming soon!!!

# Publications
<!---I think we have more than just these-->
- [Mapping Sub-Second Structure in Mouse Behavior](http://datta.hms.harvard.edu/wp-content/uploads/2018/01/pub_23.pdf)
- [The Striatum Organizes 3D Behavior via Moment-to-Moment Action Selection](http://datta.hms.harvard.edu/wp-content/uploads/2019/06/Markowitz.final_.pdf)
- [Revealing the structure of pharmacobehavioral space through motion sequencing](https://www.nature.com/articles/s41593-020-00706-3)
- [Q&A: Understanding the composition of behavior](http://datta.hms.harvard.edu/wp-content/uploads/2019/06/Datta-QA.pdf)
