import os
import shutil
from setuptools import setup
import distutils

import subprocess as sbp

from pathlib import Path


# ----- HELPER FUNCTIONS
def find_mpi_installation():
    """
    .. todo::
        * Can we get this in a correct and acceptable way from here?  How would
          we know that what we are getting is what was used to build OpenBT?
          If we build OpenBT here, then maybe we can get these with some
          certainty.
        * Is it really a good idea to presume that we can know what MPI goods
          need to be included in the installation?
    """
    PKG_CFG = "pkg-config"
    MPI_IMPL = {"mpich", "ompi"}

    if shutil.which(PKG_CFG) is None:
        raise RuntimeError("Please install pkg-config")

    pkgs_all = set()

    cmd = [PKG_CFG, "--list-all"]
    response = sbp.run(cmd, capture_output=True)
    for each in response.stdout.decode().split("\n"):
        tmp = each.split()
        if tmp:
            assert len(tmp) >= 2
            pkgs_all.add(tmp[0])
    mpi_pkgs = list(MPI_IMPL.intersection(pkgs_all))
    if not mpi_pkgs:
        raise RuntimeError("No known MPI implementations installed")
    elif len(mpi_pkgs) != 1:
        raise RuntimeError("More than one MPI implementation installed")
    pkg = mpi_pkgs[0]

    cmd = [PKG_CFG, pkg, "--variable=prefix"]
    response = sbp.run(cmd, capture_output=True)
    mpi_path = Path(response.stdout.decode().rstrip()).joinpath("bin").resolve()
    if not mpi_path.is_dir():
        msg = f"Expected {mpi_path} to be an MPI installation directory"
        raise RuntimeError(msg)
    elif not mpi_path.joinpath("mpirun").is_file():
        raise RuntimeError(f"Could not find mpirun in {mpi_path}")
    elif not mpi_path.joinpath("mpiexec").is_file():
        raise RuntimeError(f"Could not find mpiexec in {mpi_path}")

    return mpi_path

# ----- HARDCODED VALUES
# Important paths for our installation structure
CLONE_ROOT = Path(__file__).parent.resolve()
SRC_PATH = CLONE_ROOT.joinpath("src")
LIBS_PATH = SRC_PATH.joinpath(".libs")
INSTALL_PATH = CLONE_ROOT.joinpath("openbtmixing")
MPI_INSTALL_PATH = INSTALL_PATH.joinpath("mpi")
MPI_BIN_INSTALL_PATH = MPI_INSTALL_PATH.joinpath("bin")

# Hack to figure out where mpiexec is installed
MPIEXEC_PATH = find_mpi_installation()

# Names of OpenBT command line tools (CLT)
CLT_NAMES = [
    "openbtcli",
    "openbtmixing", "openbtmixingpred", "openbtmixingwts",
    "openbtmopareto",
    "openbtpred",
    "openbtsobol",
    "openbtvartivity"
]

# MPI executables that we know do not need to be included in the package.
#
# Presently, these are all just known MPI compiler wrappers.
MPI_BIN_EXCLUDE = {
    "mpicc",
    "mpic++", "mpicxx",
    "mpif77", "mpif90", "mpifort", "mpiifort",
}

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

# Copy in CLI tools (not the temporary libtool-generated wrappers)
for name in CLT_NAMES:
    shutil.copy(str(LIBS_PATH.joinpath(name)), str(INSTALL_PATH))

# Copy across most all MPI binaries since we don't know a priori which are
# needed to run mpiexec across all MPI implementations and their versions.
#
# Assume that all library external dependencies of the CLT and the MPI binaries
# will be added into wheels retroactively using a tool such as auditwheel or
# delocate to get the correct RPATHs.  Presently assuming that those will
# complete the minimal MPI installation that we are putting into the package.
rel_path = MPI_BIN_INSTALL_PATH.relative_to(INSTALL_PATH)
mpi_bins = list(MPIEXEC_PATH.glob("*"))
mpi_bins_rel_path = []
for each in mpi_bins:
    if each.name not in MPI_BIN_EXCLUDE:
        # Some symlinks might lead out of this folder.  By doing this, we 
        # assume that copying those here will not ultimately break any
        # dependencies of the target.
        shutil.copy(str(each), str(MPI_BIN_INSTALL_PATH), follow_symlinks=True)
        mpi_bins_rel_path.append(str(rel_path.joinpath(each.name)))

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
