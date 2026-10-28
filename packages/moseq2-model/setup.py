import codecs
import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="moseq2-model",
    version=get_version("moseq2_model/__init__.py"),
    author="Datta Lab",
    description="Modeling for the best",
    # "tests" is a top-level package here, so it has to be excluded explicitly
    # or it installs into site-packages under that name.
    packages=find_packages(exclude=["docs", "tests", "tests.*"]),
    include_package_data=True,
    platforms="any",
    license="LicenseRef-Harvard-Academic-Non-Commercial",
    license_files=["LICENSE.md"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    # The former ceiling of <3.8 existed for the pyhsmm/pybasicbayes/
    # autoregressive wheels, which are no longer dependencies.
    #
    # The floor is 3.10 because jax-moseq declares it and keypoint-moseq depends
    # on jax-moseq. There is no ceiling: what the upper bound should be is what
    # the version matrix measures, and a ceiling here would refuse the columns
    # that produce the answer.
    python_requires=">=3.10",
    install_requires=[
        # The model itself. 0.3.5 is the first release carrying the robust
        # AR-HMM work this package fits through: robust x separate_trans, the
        # robust evaluation surface, stateseq_marginals and stateseq_mode.
        "jax-moseq>=0.3.5",
        "jax",
        # Runtime dependencies, floors rather than pins. The previous "=="
        # pins were contemporaneous with Python 3.7 and none of them has a
        # wheel for the interpreters this now supports.
        "numpy",
        "scipy",
        "h5py>=3",
        "click>=8",
        "joblib",
        "cytoolz",
        "tqdm>=4.48.0",
        "matplotlib",
        # 0.18 removed the module-level safe_load; moseq2_model._yaml uses the
        # YAML() interface, which is available from 0.17 onward.
        "ruamel.yaml>=0.17",
    ],
    entry_points={"console_scripts": ["moseq2-model = moseq2_model.cli:cli"]},
)
