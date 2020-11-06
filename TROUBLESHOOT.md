# MoSeq2-App Troubleshooting Guide

Below is a list of common issues that users may run experience during installation, and their proposed solutions.

## Dependency Errors

### pyhsmm
If the `moseq2-model` dependency: `pyhsmm` or `pyhsmm-autoregressive` failed to install, 
 the GCC version may not be properly set.

1. Ensure acceptable version of gcc is installed:
 - Accepted versions: [6.2.0, 7.5.0, gcc-9/g++-9] 
```.bash
which gcc-7 # or `which gcc`
``` 

2. Set the environmental GCC path environment variable:
```.bash
export CC="$(which gcc-7)"
export CXX="$(which g++-7)"
``` 

3. Reinstall `moseq2-app`:
```.bash
pip install -e .
```

### moseq2
If any of the `moseq2` dependencies are missing, simply install the `moseq2-app` to install the latest
working versions of all the `moseq2` tools.
```.bash
cd moseq2-app/
pip install -e .
```

***

## Videos in Notebook not loading

If the interactive videos are not loading, ensure that the notebook is located in the parent directory
 of the dataset being extracted/analyzed.
