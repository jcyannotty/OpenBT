# Open Bayesian Trees Project
This repository includes new developments with Bayesian Additive Regression Trees and extends the original OpenBT repository created by Matt Pratola (https://bitbucket.org/mpratola/openbt/src/master/).
Such extensions include Bayesian Model Mixing and Bayesian Calibration. 
All of the Bayesian Tree code is written in C++. User interfaces constructed in R and Python allow one to easily run the software. The OpenBT project is also included in the Bayesian Model Mixing python package, *Taweret*, which is included in the Bayesian Analysis of Nuclear Dynamics sotware (see https://bandframework.github.io/). 


# Installation for Python Users

You can work with the BART-based model mixing method via the Taweret python package. Simply install Taweret and you can begin working. 

**Windows Users**: 

OpenBT relies on OpenMPI, which is not compatible with Windows. Thus you can work with Taweret by using Windows Subsystem for Linux (WSL). See instructions below for installing WSL.


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