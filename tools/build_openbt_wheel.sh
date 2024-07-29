#!/bin/bash

#
# Build the OpenBTMixing Python package as a binary wheel dedicated for
# installation in machines with compatible Python version, OS, and CPU as the
# machine doing the building.
#
# The package installs the libopenbtmixing library and associated C++ command
# line tools (CLTs), which are assumed to already exist.
#
# Users must pass the path to the root folder in which libopenbtmixing and CLTs
# are installed.  It is assumed that the binaries are in the bin subfolder.
#
# ./tools/build_openbt_wheel.sh ~/local/OpenBT
#
# The resulting wheel in is the package's dist folder.
#
# This script returns exit codes that should make it compatible with use in CI
# build processes.
#

#####----- EXTRACT BUILD INFO FROM COMMAND LINE ARGUMENT
if [[ "$#" -ne 1 ]]; then
    echo
    echo "build_openbt_wheel.sh /installation/path/OpenBT"
    echo
    exit 1
fi
prefix=$1
export CLT_BIN_INSTALL=$prefix/bin

# ----- SETUP & CHECK ENVIRONMENT
script_path=$(dirname -- "${BASH_SOURCE[0]}")
clone_root=$script_path/..
pypkg_root=$clone_root/openbtmixing_pypkg

# ----- LOG IMPORTANT DATA
echo
echo "Dependency Information"
echo "---------------------------------------------"
which mpirun
which mpicxx
mpicxx -show

echo
echo "Python version information"
echo "---------------------------------------------"
which python
python --version
echo
which pip
echo
pip list

echo
echo "Related Env Vars"
echo "---------------------------------------------"
echo "CLT_BIN_INSTALL       $CLT_BIN_INSTALL"

# ----- BUILD WHEEL
pushd $pypkg_root &> /dev/null || exit 1

echo
echo "Building Base OpenBTMixing Binary Wheel"
echo "---------------------------------------------"
# TODO: We need to build these with specific Py Version, OS, CPU in name
# Because we are building CLTs manually by hand and including in the wheels, we
# cannot distribute Python source distributions for now without including the
# C++ files and build system in the distribution and providing involved
# instructions.
python -m build --wheel || exit 1

echo
echo "Include libopenmixing in package and fix CLTs"
echo "---------------------------------------------"
delocate-listdeps --all --depending dist/openbtmixing-*.whl || exit 1

# This brings in the library and hardwires the CLTs via relative path to only
# use our library.  Therefore, an installation cannot accidentally use a
# different version of the library in a machine.
# TODO: This is for macOS binaries only
delocate-wheel -v -e libmpi dist/openbtmixing-*.whl         || exit 1

tar tvfz dist/openbtmixing-*.whl

popd &> /dev/null
