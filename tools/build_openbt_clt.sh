#!/bin/bash

#
# Build the OpenBT C++ libraries and install them in a location where Python
# package's setup can find them for installation and use.
#
# Users must pass the path to the folder in which OpenBT should be installed.
# IMPORTANT: The given folder will be removed before the build starts!
#
# ./tools/build_openbt_clt.sh ~/local/OpenBT
#
# This script returns exit codes that should make it compatible with use in CI
# build processes.
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
build_dir=$clone_root/builddir

if ! command -v mpicc &> /dev/null; then
    echo
    echo "Please install MPI with mpicc C compiler wrapper"
    echo
    exit 1
elif ! command -v mpicxx &> /dev/null; then
    echo
    echo "Please install MPI with mpicxx C++ compiler wrapper"
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

echo
echo "CC=$CC"
echo "CXX=$CXX"
echo "MPICC=$MPICC"
echo "MPICXX=$MPICXX"
echo "CPATH=$CPATH"
echo "CFLAGS=$CFLAGS"
echo "CXXFLAGS=$CXXFLAGS"
echo "CPPFLAGS=$CPPFLAGS"
echo "LDFLAGS=$LDFLAGS"
echo

if ! command -v meson &> /dev/null; then
    echo
    echo "Please install meson"
    echo
    exit 1
fi

# ----- LOG IMPORTANT DATA
echo
echo "meson version information"
echo "---------------------------------------------"
meson --version

# ----- CLEAN-UP LEFTOVERS FROM PREVIOUS BUILDS

echo
echo "Clean-up build environment"
echo "---------------------------------------------"
rm -rf $prefix
rm -rf $build_dir

popd &> /dev/null

# ----- SETUP BUILD SYSTEM, CONFIGURE, & BUILD
pushd $clone_root &> /dev/null || exit 1

echo
echo "Configure OpenBT"
echo "---------------------------------------------"
meson setup --wipe --buildtype=release $build_dir -Dprefix=$prefix -Dmpi=open-mpi -Dverbose=false || exit 1

echo
echo "Make & Install OpenBT"
echo "---------------------------------------------"
meson compile -v -C $build_dir          || exit 1
meson install --quiet -C $build_dir     || exit 1

popd &> /dev/null
