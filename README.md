<!-- [![Build Status](https://app.travis-ci.com/dattalab/moseq2-app.svg?token=ptXpSa3Fp9PKqkxJDkDr&branch=dev)](https://app.travis-ci.com/dattalab/moseq2-app) -->
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1oc0_0mlN0VTZEPMQTg_hnYAC87Lb58MI" />
</p>

Welcome to the `moseq2-app` repository. Motion Sequencing (MoSeq) is an unsupervised machine learning method used to describe mouse behavior and `moseq2-app` is the starting point to MoSeq2 package suite.

 To get started, head over to the [wiki](https://github.com/dattalab/moseq2-app/wiki) to find which installation option works best for your environment and detailed documentation for the MoSeq2 package suite.

**Quick Links:**
[![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WV97_Ko7qu6-l8sE03DCG6SRxcua_3eX?usp=sharing)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Wiki-blue)](https://github.com/dattalab/moseq2-app/wiki)
[![DOI](https://zenodo.org/badge/204555901.svg)](https://zenodo.org/badge/latestdoi/204555901)


Last Updated: 12/23/2021

# Overview

<!---Adapted from 
Datta, Sandeep Robert, David J. Anderson, Kristin Branson, Pietro Perona, and Andrew Leifer. 2019. “Computational Neuroethology: A Call to Action.” Neuron 104 (1): 11–24.
-->
**MoSeq takes 3D depth videos as input (obtained using commercially-available sensors) and then uses statistical learning techniques to identify the components of mouse body language.** This is achieved by fitting an autoregressive hidden Markov model that parses behavior into a set of sub-second motifs called syllables. This segmentation naturally yields boundaries between syllables, and therefore also reveals the structure that governs the interconnections between syllables over time, which we refer to as behavioral grammar (see [Wiltschko et al., 2015](https://www.sciencedirect.com/science/article/pii/S0896627315010375) for the first description of MoSeq). 

Because MoSeq relies on unsupervised machine learning, MoSeq discovers the set of syllables and grammar expressed in any given experiment. By combining MoSeq with electrophysiology, multi-color photometry, and miniscope methods, neural correlates for 3D behavioral syllables have recently been identified in the dorsolateral striatum (DLS) ([Markowitz et al., 2018](https://www.sciencedirect.com/science/article/pii/S0092867418305129)). Furthermore, MoSeq has been combined with optogenetic stimulation to reveal the differential consequences of activating the motor cortex, the dorsal striatum, and the ventral striatum ([Pisanello et al.,2017](https://www.nature.com/articles/nn.4591); [Wiltschko et al., 2015](https://www.sciencedirect.com/science/article/pii/S0896627315010375)). These results are consistent with similar results recently obtained using marker-based approaches to explore the relationship between 3D posture and activity in the posterior parietal cortex ([Mimica et al., 2018](https://www.science.org/doi/10.1126/science.aau2013)).

There are two basic steps to MoSeq, and this GitHub repository and the wiki supports both. First, 3D data need to be acquired. Here you will find instructions for assembling a standard MoSeq data acquisition platform, and code to acquire 3D videos of mice as they freely behave. Second, these 3D data need to be modeled. We provide several different methods for modeling the data using MoSeq, which are compared below. We continue development of MoSeq and plan to incorporate additional features in the near future. 

## MoSeq2 package suite
The MoSeq2 toolkit enables users to model mouse behavior across different experimental groups, and measure the differences between their behavior usages, durations, transition patterns. etc.

This package contains functionalities that can be used interactively in jupyter notebooks. 
We provide a series of Jupyter Notebooks that cover the entire MoSeq pipeline to process their depth videos of mice, and segment their behavior into what is denoted as "syllables". In addition to the Jupyter notebooks, MoSeq has Google Colab notebooks and a Command Line Interface. Consult the wiki page for more detailed documentation of the MoSeq pipeline [here](https://github.com/dattalab/moseq2-app/wiki).
You can try MoSeq on [Google Colab](https://colab.research.google.com/drive/1WV97_Ko7qu6-l8sE03DCG6SRxcua_3eX?usp=sharing) on our test data or your data on Google Drive.

# Getting Started

If you like MoSeq and you are interested in installing it in your environment, you can install the MoSeq2 package suite with either Conda or Docker. 

- If you are familiar with Conda/terminal, and you enjoy more control over the packages and virtual environment, we recommend installing the MoSeq2 package suite with Conda.

- If you are interested in using a standardized/containerized version of the MoSeq2 package suite with simple installation steps and minimum local environment setup, we recommend installing the MoSeq2 package suite with Docker.

### **We provide step-by-step guides for installing the MoSeq2 package suite in the [installation documentation](https://github.com/dattalab/moseq2-app/wiki/MoSeq2-Installation).**

You can find more information about the MoSeq2 package suite, the acquisition and analysis pipeline, documentation for Command Line Interface (CLI), tutorials, etc. in the [wiki](https://github.com/dattalab/moseq2-app/wiki). If you want to explore MoSeq functionalities, check out the guide for [downloading test data](https://github.com/dattalab/moseq2-app/wiki/Download-Test-Data).

# Community Support and Contributing
- Please join [![MoSeq Slack Channel](https://img.shields.io/badge/slack-MoSeq-blue.svg?logo=slack)](https://moseqworkspace.slack.com) to post questions and interactive with MoSeq developers and users.
- If you encounter bugs, errors or issues, please submit a Bug report [here](https://github.com/dattalab/moseq2-app/issues/new/choose). We encourage you to check out the [troubleshooting and tips section](https://github.com/dattalab/moseq2-app/wiki/Troubleshooting-and-Tips) and search your issues in [the existing issues](https://github.com/dattalab/moseq2-app/issues) first.   
- If you want to see certain features in MoSeq or you have new ideas, please submit a Feature request [here](https://github.com/dattalab/moseq2-app/issues/new/choose).
- If you want to contribute to our codebases, please check out our [Developer Guidelines](https://github.com/dattalab/moseq2-app/wiki/MoSeq-Developer-Guidelines).
- Please tell us what you think by filling out [this user survey](https://forms.gle/S88jptAEs41mQjff7).

# Versions
- Current version: Version 1.2.0
- [Changelog](https://github.com/dattalab/moseq2-app/wiki/Changelog)

# Events & News
<!---Future events, project related news etc-->
We are hosting a tutorial workshop on Thursday, March 3rd, 2022 at 1:30-4PM EST.
Fill out [this form](https://forms.gle/gp2ipZ6BTyFf1GrA8)
by March 2nd to register for the workshop and receive the Zoom link and password.

# Publications
- [Mapping Sub-Second Structure in Mouse Behavior](http://datta.hms.harvard.edu/wp-content/uploads/2018/01/pub_23.pdf)
- [The Striatum Organizes 3D Behavior via Moment-to-Moment Action Selection](http://datta.hms.harvard.edu/wp-content/uploads/2019/06/Markowitz.final_.pdf)
- [Q&A: Understanding the composition of behavior](http://datta.hms.harvard.edu/wp-content/uploads/2019/06/Datta-QA.pdf)
- [Computational Neuroethology: A Call to Action]("http://datta.hms.harvard.edu/wp-content/uploads/2020/01/AC2A.pdf")
- [Revealing the structure of pharmacobehavioral space through motion sequencing](https://www.nature.com/articles/s41593-020-00706-3)

# License

MoSeq is freely available for academic use under a license provided by Harvard University. Please refer to the license file for details. If you are interested in using MoSeq for commercial purposes please contact Bob Datta directly at srdatta@hms.harvard.edu, who will put you in touch with the appropriate people in the Harvard Technology Transfer office.
