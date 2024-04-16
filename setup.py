import os
import shutil
from setuptools import setup
import distutils

from pathlib import Path


# ----- HARDCODED VALUES
CLONE_ROOT = Path(__file__).parent.resolve()
SRC_PATH = CLONE_ROOT.joinpath("src")
LIBS_PATH = SRC_PATH.joinpath(".libs")
INSTALL_PATH = CLONE_ROOT.joinpath("openbtmixing")
MPI_INSTALL_PATH = INSTALL_PATH.joinpath("mpi")
MPI_BIN_INSTALL_PATH = MPI_INSTALL_PATH.joinpath("bin")

# Names of OpenBT command line tools (CLT)
CLT_NAMES = [
    "openbtcli",
    "openbtmixing", "openbtmixingpred", "openbtmixingwts",
    "openbtmopareto",
    "openbtpred",
    "openbtsobol",
    "openbtvartivity"
]

# ----- COPY PRE-BUILT C++ OpenBT RESULTS INTO INSTALL PATH
# Always start with a clean installation folder
if MPI_INSTALL_PATH.exists():
    assert MPI_INSTALL_PATH.is_dir()
    shutil.rmtree(MPI_INSTALL_PATH)
os.mkdir(MPI_INSTALL_PATH)
os.mkdir(MPI_BIN_INSTALL_PATH)

for name in CLT_NAMES:
    clt_path = INSTALL_PATH.joinpath(name)
    if clt_path.exists():
        assert clt_path.is_file()
        os.remove(str(clt_path))

# Copy in CLI programs (not the temporary libtool-generated wrappers)
for name in CLT_NAMES:
    shutil.copy(str(LIBS_PATH.joinpath(name)), str(INSTALL_PATH))

# Copy across all MPI binaries since we don't know a priori across all MPI
# implementations and their versions which are needed to run mpiexec.
#
# Assume that all library external dependencies of the CLT and the MPI binaries
# will be added into wheels retroactively using a tool such as auditwheel or
# delocate to get the correct RPATHs.  Presently assuming that those will
# complete the minimal MPI installation that we are putting into the package.
#
# TODO: Can we get this in a correct and acceptable way from here?  How would
# we know that what we are getting is what was used to build OpenBT?  If we
# build OpenBT here, then maybe we can get these with some certainty.
# TODO: Is it really a good idea to presume that we can know what MPI goods
# need to be included in the installation?
MPICC_PATH = Path(shutil.which("mpicc")).parent.resolve()
mpi_bins = list(MPICC_PATH.glob("*"))
for each in mpi_bins:
    # TODO: Is this preserving or resolving symlinks?
    shutil.copy(str(each), str(MPI_BIN_INSTALL_PATH))

rel_path = MPI_BIN_INSTALL_PATH.relative_to(INSTALL_PATH)
mpi_bins_rel_path = [str(rel_path.joinpath(e)) for e in mpi_bins]

# Setup
dist_name = distutils.util.get_platform()
dist_name = dist_name.replace("-","_")
dist_name = dist_name.replace(".","_")

if "linux_x86_64" in dist_name:
    dist_name = "manylinux2014_x86_64"

setup(
    name='openbtmixing',
    version='1.0.0',
    packages=["openbtmixing"],
    package_data={'openbtmixing': CLT_NAMES + mpi_bins_rel_path},
    zip_safe=False,
    options={'bdist_wheel':{'plat_name':dist_name}}
)
