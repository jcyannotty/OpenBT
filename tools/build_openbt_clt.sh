#!/bin/bash

#
# Build the OpenBT C++ libraries and install them in a location where Python
# package's setup can find them for installation and use.
#
# Users must pass the path to the folder in which OpenBT should be installed.
# IMPORTANT: The given folder will be removed as part of the build!
#
# On some systems I have needed to run as
#
# CPATH=~/local/eigen ./tools/build_openbt_clt.sh ~/local/OpenBT
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
    echo "build_openbt_clt.sh /installation/path/OpenBT"
    echo
    exit 1
fi
prefix=$1

# ----- SETUP & CHECK ENVIRONMENT
script_path=$(dirname -- "${BASH_SOURCE[0]}")
clone_root=$script_path/..

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

echo
echo "MPI wrappers"
echo "---------------------------------------------"
which mpicc
mpicc -show
mpicc --version
echo
which mpicxx
mpicxx -show
mpicxx --version
echo

# These are required by configure on my system
# - macOS with OpenMPI installed via homebrew
export CC=$(which mpicc)
export CXX=$(which mpicxx)

echo
echo "CC=$CC"
echo "CXX=$CXX"
echo "MPICC=$MPICC"
echo "MPICXX=$MPICXX"
echo "LDFLAGS=$LDFLAGS"
echo "CPATH=$CPATH"
echo "LIBRARY_PATH=$LIBRARY_PATH"
echo

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

# ----- CLEAN-UP LEFTOVERS FROM PREVIOUS BUILDS
pushd $clone_root &> /dev/null || exit 1

echo
echo "Clean-up build environment"
echo "---------------------------------------------"
rm -rf $prefix

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
./configure --with-mpi --with-silent --prefix=$prefix || { cat config.log; exit 1; }

# We need to install the libraries so that they are in the location provided in
# the RPATH of the CLI programs.  Then tools like auditwheel and delocate can
# find them and include them in the wheel.
echo
echo "Make & Install OpenBT"
echo "---------------------------------------------"
make clean install || exit 1

popd &> /dev/null
