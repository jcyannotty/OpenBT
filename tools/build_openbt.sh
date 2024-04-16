#!/bin/bash

#
# Build the OpenBT C++ libraries and a Python binary wheel that uses the
# library.  The wheel contains programs with dependencies set with RPATH values
# that will not work with wheels.  They should be fixed appropriately with
# tools like auditwheel and delocate.
#
# Users must pass the path to the folder in which OpenBT should be installed.
# IMPORTANT: The given folder will be removed as part of the build!
#
# On some systems I have needed to run as
#
# CPATH=~/local/eigen ./tools/build_openbt.sh ~/local/OpenBT
#
# in order for configuration to run through properly.
#
# TODO: See if we can get configure to find Eigen directly or by using
# pkg-config so that I don't have to set CPATH.
# TODO: See if we can get configure to use MPI wrappers rather than C/C++
# compilers to build so that I don't have to set CC/CXX to wrappers.
#

#####----- EXTRACT BUILD INFO FROM COMMAND LINE ARGUMENT
if [[ "$#" -ne 1 ]]; then
    echo
    echo "build_opebt.sh /installation/path/OpenBT"
    echo
    exit 1
fi
prefix=$1

# ----- SETUP & CHECK ENVIRONMENT
script_path=$(dirname -- "${BASH_SOURCE[0]}")
clone_root=$script_path/..

if ! command -v autoconf &> /dev/null; then
    echo
    echo "Please install autoconf"
    echo
    exit 1
elif ! command -v automake &> /dev/null; then
    echo
    echo "Please install automake"
    echo
    exit 1
elif ! command -v aclocal &> /dev/null; then
    echo
    echo "Check if aclocal installed with automake"
    echo
    exit 1
fi

if command -v libtoolize &> /dev/null; then
    libtoolize_exe=libtoolize
elif command -v glibtoolize &> /dev/null; then
    libtoolize_exe=glibtoolize
else
    echo
    echo "Unable to locate either libtoolize or glibtoolize"
    echo
    exit 1
fi

if ! command -v mpicc &> /dev/null; then
    echo
    echo "Please install MPI with mpicc C compiler wrapper"
    echo
    exit 1
elif ! command -v mpic++ &> /dev/null; then
    echo
    echo "Please install MPI with mpic++ C++ compiler wrapper"
    echo
    exit 1
fi
# These are required by configure on my system
# - macOS with MPICH installed via homebrew
export CC=$(which mpicc)
export CXX=$(which mpicxx)

# ----- LOG IMPORTANT DATA
echo
echo "Autotools version information"
echo "---------------------------------------------"
autoconf --version
echo
automake --version
echo
aclocal  --version
echo
$libtoolize_exe --version

echo
echo "MPI wrappers"
echo "---------------------------------------------"
echo "CC=$CC"
echo "CXX=$CXX"
echo
which mpicc
mpicc -show
echo
which mpicxx
mpicxx -show
echo

# ----- CLEAN-UP LEFTOVERS FROM PREVIOUS BUILDS
pushd $clone_root/src &> /dev/null || exit 1

echo
echo "Clean-up build environment"
echo "---------------------------------------------"
rm -rf $prefix
rm -rf build
rm -rf openbtmixing.egg-info

popd &> /dev/null

# ----- SETUP BUILD SYSTEM, CONFIGURE, & BUILD
pushd $clone_root/src &> /dev/null || exit 1

echo
echo "Setting up build system"
echo "---------------------------------------------"
$libtoolize_exe        || exit 1
aclocal                || exit 1
automake --add-missing || exit 1
autoconf               || exit 1

echo
echo "Configure OpenBT"
echo "---------------------------------------------"
./configure --with-mpi --with-silent --prefix=$prefix || exit 1

# We need to install the libraries so that they are in the location provided in
# the RPATH of the CLI programs.  Then tools like auditwheel and delocate can
# find them and include them in the wheel.
echo
echo "Make & Install OpenBT"
echo "---------------------------------------------"
make clean install || exit 1

popd &> /dev/null

# ----- BUILD PYTHON BINARY WHEEL
pushd $clone_root &> /dev/null

echo
echo "Build binary wheel"
echo "---------------------------------------------"
pip wheel . || exit 1
echo

popd &> /dev/null
