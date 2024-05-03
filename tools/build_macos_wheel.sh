#!/bin/bash

#
# Build a Python binary wheel for macOS that contains and uses the OpenBT C++
# command line tools (CLTs) and associated libraries.  The script uses delocate
# to include the libraries in the wheel and setup the CLTs so that they are
# hardcoded to use the libraries in the wheel via relative path, which should
# be compatible with arbitrary pip installs on other machines.
#
# At present, the package does not contain any MPI binaries or libraries.
# Rather, it assumes that users will have already installed an OpenMPI
# implementation that is compatible with the implementation used to build the
# package and that the package will be able to find these automatically.
#
# Note that this script removes all openbtmixing-*.whl files in the current
# working directory and all *.whl files in the openbtmixing_pypkg folder before
# it builds the wheel.
#

#####----- EXTRACT BUILD INFO FROM COMMAND LINE ARGUMENT
if [[ "$#" -ne 0 ]]; then
    echo
    echo "build_macos_wheel.sh"
    echo
    exit 1
fi

script_path=$(dirname -- "${BASH_SOURCE[0]}")
clone_root=$script_path/..
pypkg_path=$clone_root/openbtmixing_pypkg

echo
echo "Clean-up $pypkg_path"
echo "---------------------------------------------"
rm openbtmixing-*.whl

pushd $pypkg_path &> /dev/null
ls -la
cat VERSION

rm *.whl

echo
echo "Build binary wheel"
echo "---------------------------------------------"
pip wheel . || exit 1
echo

echo
echo "delocate binary wheel"
echo "---------------------------------------------"
delocate-listdeps --all openbtmixing-*.whl
echo
# This presupposes that we know what all the MPI-related external dependencies
# are => human in the loop that reviews this output.
delocate-wheel -e open-mpi -e libevent -e hwloc -e pmix -v openbtmixing-*.whl
echo
delocate-listdeps --all openbtmixing-*.whl
echo
popd &> /dev/null

mv $pypkg_path/openbtmixing-*.whl .
rm $pypkg_path/*.whl

tar tvfz openbtmixing-*.whl
echo
