#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo
    echo "Please pass GitHub action runner OS (e.g., Linux or macOS)"
    echo
    exit 1
fi
runner_os=$1

which python
which pip
echo " "
python -c "import platform ; print(platform.machine())"
python -c "import platform ; print(platform.system())"
python -c "import platform ; print(platform.release())"
python -c "import platform ; print(platform.platform())"
python -c "import platform ; print(platform.version())"
if   [ "$runner_os" = "macOS" ]; then
    python -c "import platform ; print(platform.mac_ver())"
    echo " "
    # Get information on number of processors, which sets limit on number of
    # "slots" available for MPI processes in OpenMPI.
    sysctl hw.ncpu
    sysctl hw.physicalcpu
    sysctl hw.logicalcpu
elif [ "$runner_os" = "Linux" ]; then
    echo " "
    # Get information on number of cores & hardware threads, which sets limit
    # on number of "slots" available for MPI processes in OpenMPI.
    lscpu
else
    echo
    echo "Invalid runner OS $runner_os"
    echo
    exit 1
fi
echo " "
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
python -m pip install build
python -m pip install tox
python -m pip install --user meson
echo " "
python --version
tox --version
meson --version
echo " "
pip list
echo " "
