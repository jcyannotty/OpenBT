#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo
    echo "install_meson.sh /installation/path {macOS, Linux}"
    echo
    exit 1
fi
install_path=$1
runner_os=$2

venv_path=$install_path/local/venv
meson_venv=$venv_path/meson
local_bin=$install_path/local/bin

if   [ "$runner_os" = "macOS" ]; then
    brew install ninja
elif [ "$runner_os" = "Linux" ]; then
    sudo apt-get update
    sudo apt-get -y install ninja-build
else
    echo
    echo "Invalid runner OS $runner_os"
    echo
    exit 1
fi

echo " "
mkdir -p $venv_path
mkdir -p $local_bin

# Beginning with v1.6.0 meson can automatically find OpenMPI and MPICH
python -m venv $meson_venv
. $meson_venv/bin/activate
which python
which pip
python -m pip install --upgrade pip
python -m pip install meson>=1.6.0
echo " "
python -m pip list
echo " "

# Install just meson command in path
ln -s $meson_venv/bin/meson $local_bin
echo "$local_bin" >> "$GITHUB_PATH"
echo " "
