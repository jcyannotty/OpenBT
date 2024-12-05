# Open Bayesian Trees Project
This repository includes new developments with Bayesian Additive Regression Trees and extends the original OpenBT repository created by Matt Pratola (https://bitbucket.org/mpratola/openbt/src/master/).
Such extensions include Bayesian Model Mixing and Bayesian Calibration. 
All of the Bayesian Tree code is written in C++. User interfaces constructed in R and Python allow one to easily run the software. The OpenBT project is also included in the Bayesian Model Mixing python package, *Taweret*, which is included in the Bayesian Analysis of Nuclear Dynamics software (see https://bandframework.github.io/). 


# Installation
The heart of OpenBTMixing is a set of C++ command line tools, which are wrapped
by the Python and R packages.  Typically these tools are built with MPI to
enable distributed parallelization of computations.  In particular, the Python
wrapper package is always built with MPI support.

The software and its distribution scheme have been developed to allow users to
use OpenBTMixing with the MPI installation of their choosing.  For instance, it
can be built with MPI installations on leadership class platforms and
clusters that were installed by experts and optimized for the platform.  As a
result, however, the software cannot be distributed as prebuilt binaries or
wheels, but rather must be built for each case with the compiler suite and
matching MPI implementation provided by the user.

## Requirements
Before building and installing the bare C++ tools or the Python package, users
must provide
* a compiler suite that includes a C++ compiler,
* an MPI installation that is compatible with the compiler suite,
* the [Meson build system](https://mesonbuild.com) and its prerequistes such as
  [ninja](https://ninja-build.org), and
* optionally the [Eigen software package](https://gitlab.com/libeigen/eigen).

The Meson build system is setup to automatically detect the compiler suite and
MPI installation to use.  If Eigen already exists in the system and Meson can
find it, then it will use it for the build.  Otherwise, it will automatically
obtain a copy of Eigen for internal use.

We presently test OpenBTMixing with both [Open MPI](https://www.open-mpi.org)
and [MPICH](https://www.mpich.org) implementations and have successfully used
the Python package using MPI implementations installed
* via package managers such as Ubuntu's Advanced Packaging Tool (`apt`) and
  `homebrew` on macOS;
* by experts on, for example, clusters and available as modules; and
* with `conda` from the prebuilt conda forge
  [openmpi](https://anaconda.org/conda-forge/openmpi) package.

## Python package
The OpenBTMixing Python package is distributed on PyPI as a source distribution
that contains the C++ code and files needed by Meson to build the dedicated,
standalone command line tools that the package will use.  The tools are built
and installed automatically by Meson as part of executing
```
python -m pip install openbtmixing
```
The package can also be built and installed from a clone of this repository with
```
$ cd /path/to/OpenBT/openbtmixing_pypkg
$ python -m build --sdist
$ python -m pip install -v dist/openbtmixing-<version>.tar.gz
```

Developers can setup a virtual environment with a developer/editable mode
installation of the package with
```
$ /path/to/target/python -m venv /path/to/venv/my_venv
$ . /path/to/venv/my_venv/bin/activate
$ cd /path/to/OpenBT/openbtmixing_pypkg
$ python -m pip install -v -e .
```
In this latter case, the command line tools are built automatically and
installed at `/path/to/OpenBT/openbtmixing_pypkg/src/openbtmixing/bin`.  The
Python package is hardcoded to use those tools so that the existence of another
set of tools in the system and in the PATH should not cause issues.

## C++ library & command line tool interface
Developers and C++ users can directly build and install the command line tools,
an OpenBTMixing library, and library tests with `tools/build_openbt_clt.sh`.
Note that these do **not** need to be built in order to use the Python package.

## R package
**TODO**: Needs to be written based on original content of README.

## Meson issues
The Meson build system installation documents suggest installing Meson via
package manager where possible.  If it is not available or the latest available
version is too old, the following has been used to install Meson with python
into a dedicated virtual environment.

```
$ /path/to/target/python -m venv ~/local/venv/meson
$ . ~/local/venv/meson/bin/activate
$ which python
$ python -m pip install --upgrade pip
$ python -m pip install meson
$ python -m pip list
$ ln -s ~/local/venv/meson/bin/meson ~/local/bin
<append ~/local/bin to PATH if desired and appropriate>
$ which meson
$ meson --version
```

# Examples

The examples from the article "Model Mixing Using Bayesian Additive Regression Tress" are reproduced in the jupyter noteboook BART_BMM_Technometrics.ipynb. This notebook can be run locally or in a virtual environment such as google colab.


# Related Software

The BART Model Mixing software has been implemented in the [Taweret](https://github.com/TaweretOrg/Taweret/tree/main) Python package in conjunction with the [BAND](https://bandframework.github.io/) collaboration.
