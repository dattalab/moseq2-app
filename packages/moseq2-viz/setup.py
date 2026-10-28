from setuptools import setup, find_packages
import codecs
import os


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='moseq2-viz',
    author='Datta Lab',
    description='To boldly go where no mouse has gone before',
    version=get_version("moseq2_viz/__init__.py"),
    # "tests" is a top-level package here, so without an exclude it installs
    # into site-packages under that name.
    packages=find_packages(exclude=['docs', 'tests', 'tests.*']),
    platforms=['mac', 'unix'],
    license='LicenseRef-Harvard-Academic-Non-Commercial',
    license_files=['LICENSE.md'],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    # The former ceiling of <3.8 came from the pinned dependencies below, none
    # of which had a wheel past 3.7. Nothing in the source needs it.
    #
    # The floor is 3.10 because jax-moseq declares it and keypoint-moseq depends
    # on jax-moseq. There is no ceiling: what the upper bound should be is what
    # the version matrix measures, and a ceiling here would refuse the columns
    # that produce the answer.
    python_requires='>=3.10',
    install_requires=[
        # Floors, not pins. A version is named only where the code calls an API
        # that requires it; the rest are left to the resolver, since every
        # previous "==" pin was contemporaneous with Python 3.7 and none of them
        # builds on a supported interpreter.
        # ffmpeg's own libraries, as a wheel. Crowd movies are written
        # through these rather than through an ffmpeg a user has to install
        # themselves.
        'av>=17.1',
        'h5py>=3',
        # ruamel dropped the module-level safe_load/safe_dump in 0.18;
        # moseq2_viz._yaml uses the YAML() interface, present from 0.17.
        'ruamel.yaml>=0.17',
        # scipy.stats.mode grew keepdims in 1.9, which viz.py passes so the
        # result keeps the shape it had before the 1.11 default flip.
        'scipy>=1.9',
        # pointplot took linestyle in 0.13, replacing the deprecated join.
        'seaborn>=0.13',
        'click',
        'cytoolz',
        'dtaidistance',
        'joblib',
        'matplotlib',
        'networkx',
        'numpy',
        'opencv-python',
        'pandas',
        # Not imported directly; pandas needs it for the to_parquet path in
        # helpers/wrappers.py.
        'pyarrow',
        'scikit-learn',
        # Imported by model/stat.py, and missing from this list until now, so a
        # clean install could not import moseq2_viz.model.stat.
        'statsmodels',
        'tqdm',
    ],
    extras_require={
        # helpers/wrappers.py imports this lazily, behind a guarded try/except,
        # only for graphviz transition-graph layouts. It needs system graphviz
        # headers to build, so it stays opt-in.
        'graphviz': ['pygraphviz'],
    },
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)
