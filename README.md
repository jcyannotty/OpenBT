# Open Bayesian Trees Project
This repository includes new developments with Bayesian Additive Regression Trees and extends the original OpenBT repository created by Matt Pratola (https://bitbucket.org/mpratola/openbt/src/master/).
Such extensions include Bayesian Model Mixing and Bayesian Calibration. 
All of the Bayesian Tree code is written in C++. User interfaces constructed in R and Python allow one to easily run the software. The OpenBT project is also included in the Bayesian Model Mixing python package, *Taweret*, which is included in the Bayesian Analysis of Nuclear Dynamics sotware (see https://bandframework.github.io/). 


# Installation for Python Users

You can work with the BART-based model mixing method via the Taweret python package. Simply install Taweret and you can begin working. 

**Windows Users**: 

OpenBT relies on OpenMPI, which is not compatible with Windows. Thus you can work with Taweret by using Windows Subsystem for Linux (WSL). See instructions below for installing WSL.

**macOS Users**: 

There is currently no wheel in PyPI for macs with ARM processors.  While we intend
to have a permanent solution to this issue soon (See Issue #6), at present
affected users must manually build and install the package using the
procedure given here.

We assume the use of the [homebrew package manager](https://brew.sh).  For
ARM-based macs, homebrew installs packages in `/opt/homebrew`.  Please adapt
appropriately the following if you are installing by other means or if homebrew
installs to a different location.

Preinstall requirements:
* Install homebrew if not already done so
* `brew install open-mpi`
* `brew install autoconf`
* `brew install autoconf-archive`
* `brew install automake`
* `brew install libtool`
* `brew install eigen`

Confirm that installation is valid:
* Run `which mpirun` and confirm that `mpirun` is found and located in a
  reasonable location (e.g., `/opt/homebrew/bin/mpirun`).
* Run `ls /opt/homebrew/include/eigen3/Eigen` and confirm that you see an
  installation by confirming that folders such as `QR`, `SVD`, and `Dense` are
  shown.

Build the openbtmixing command line tools (CLTs):
* Clone the [openbtmixing repository](https://github.com/jcyannotty/OpenBT) on
  the machine that requires the installation
* `cd /path/to/OpenBT`
* `mkdir ~/local/OpenBT`
* `CPATH=/opt/homebrew/include/eigen3 ./tools/build_openbt_clt.sh ~/local/OpenBT`
* Run `ls ~/local/OpenBT/bin` and confirm that you see `openbtcli` and similar
* Run `ls ~/local/OpenBT/lib` and confirm that you see `libtree.dylib` and similar
* Add location of the CLTs to `PATH` with something like
  `export PATH=$PATH:$HOME/local/OpenBT/bin`
* Run `which openbtcli` and confirm that the `openbtcli` CLT is found and installed
  in the expected location

Note that users might want to add the alteration of `PATH` to a shell
configuration file such as `.zshrc` so that it is automatically setup when the
shell is started.

To build and install the package, please first setup a virtual environment with
your target Python and activate the environment.
* `cd /path/to/OpenBT/openbtmixing_pypkg`
* `python -m pip install --upgrade pip`
* `python -m pip install build`
* `python -m build --sdist`
* `python -m pip install dist/openbtmixing-<version>.tar.gz`
* `python -m pip list`
* Run `/path/to/OpenBT/tools/test_python_installation.py` and confirm that all tests are
  passing.

# Installation for R Users:

The Trees module is a Python interface which calls and executes a Ubuntu package in order to fit the mixing model and obtain the resulting predictions. This package is developed as a part of the Open Bayesian Trees Project (OpenBT). To install the Ubuntu package, please follow the steps below based on the operating system of choice.


**Linux:**

1. Download the OpenBT Ubuntu Linux 20.04 package:

```    
    $ wget -q https://github.com/jcyannotty/OpenBT/raw/main/openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb 
```    

2. Install the package and reset the library cache:

```    
    $ cd /location/of/downloaded/.deb
    $ dpkg -i openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb
    $ ldconfig

```

**Mac OS/:X**

1. Install the OS/X OpenMPI package by running the following `brew` commands in a terminal window:

```
    $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    $ brew install open-mpi
```    


2. Download the OpenBT OSX binary package: "OpenBT-Mixing-0.current.pkg".

3. Install the OpenBT OSX package by double-clicking on the downloaded .pkg file and follow the on-screen instructions.


**Windows:**

OpenBT will run within the Windows 10 Windows Subsystem for Linux (WSL) environment. For instructions on installing WSL, please see [Ubuntu WSL](https://ubuntu.com/wsl). We recommend installing the Ubuntu 20.04 WSL build. 
There are also instructions [here](https://wiki.ubuntu.com/WSL?action=subscribe&_ga=2.237944261.411635877.1601405226-783048612.1601405226#Installing_Packages_on_Ubuntu) on keeping your Ubuntu WSL up to date, or installing additional features like X support. Once you have installed the WSL Ubuntu layer, start the WSL Ubuntu shell from the start menu and then install the package:

```    
    $ cd /mnt/c/location/of/downloaded/.deb
    $ dpkg -i openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb

```    

# Local Compilation

Alternatively, one could also download the source files and compile the project locally. If compiling locally, please ensure you have installed the approriate dependencies for MPI (see Mac OS/X above) and the [Eigen Library](https://eigen.tuxfamily.org/index.php?title=Main_Page). 


# Examples

The examples from the article "Model Mixing Using Bayesian Additive Regression Tress" are reproduced in the jupyter noteboook BART_BMM_Technometrics.ipynb. This notebook can be run locally or in a virtual environment such as google colab.


# Related Software

The BART Model Mixing software has been implemented in the [Taweret](https://github.com/TaweretOrg/Taweret/tree/main) Python package in conjunction with the [BAND](https://bandframework.github.io/) collaboration.
