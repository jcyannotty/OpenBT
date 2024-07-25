import os
import shutil
import codecs

from pathlib import Path
from setuptools import setup

# ----- HARDCODED VALUES
PKG_ROOT = Path(__file__).resolve().parent
SRC_PATH = PKG_ROOT.joinpath("src", "openbtmixing")
BIN_PATH = SRC_PATH.joinpath(".bin")
if "CLT_BIN_INSTALL" not in os.environ:
    msg = "Please set CLT_BIN_INSTALL to the location of the OpenBT binaries"
    raise RuntimeError(msg)
CLT_BIN_INSTALL = Path(os.environ["CLT_BIN_INSTALL"]).resolve()

# Names of C++ products to include
CLT_NAMES = [
    "openbtcli",
    "openbtmixing", "openbtmixingpred", "openbtmixingwts",
    "openbtmopareto",
    "openbtpred",
    "openbtsobol",
    "openbtvartivity"
]

PYTHON_REQUIRES = ">=3.9"
CODE_REQUIRES = ["numpy", "matplotlib"]
TEST_REQUIRES = ["pytest", "scipy", "pandas"]
INSTALL_REQUIRES = CODE_REQUIRES + TEST_REQUIRES

# TODO: Try to use BIN_PATH here if possible
PACKAGE_DATA = {
    'openbtmixing':
        ["tests/bart_bmm_test_data/2d_*.txt", ".bin/*"]
}

PROJECT_URLS = {
    "Source": "https://github.com/jcyannotty/OpenBT",
    "Documentation": "https://github.com/jcyannotty/OpenBT",
    "Tracker": "https://github.com/jcyannotty/OpenBT/issues",
}


# ----- SPECIFY THE PACKAGE
def readme_md():
    fname = PKG_ROOT.joinpath("README.md")
    with codecs.open(fname, "r", encoding="utf8") as fptr:
        return fptr.read()


def version():
    fname = PKG_ROOT.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()

# ----- INCLUDE COMMAND LINE TOOLS
# Always start with a clean installation folder
if BIN_PATH.exists():
    shutil.rmtree(BIN_PATH)
os.mkdir(BIN_PATH)

# Only include the CLTs.  To finalize wheels, we must use delocate or
# auditwheel to pull in *only* the OpenBT library and *no* MPI-related
# dependencies.  This will fix the CLTs so that they use only the library
# installed with them via a relative path.
for name in CLT_NAMES:
    shutil.copy(str(CLT_BIN_INSTALL.joinpath(name)), str(BIN_PATH))

setup(
    name='openbtmixing',
    version=version(),
    author="John Yannotty",
    author_email="yannotty.1@buckeyemail.osu.edu",
    maintainer="John Yannotty",
    maintainer_email="yannotty.1@buckeyemail.osu.edu",
    license="MIT",
    package_dir={"": "src"},
    package_data=PACKAGE_DATA,
    url=PROJECT_URLS["Source"],
    project_urls=PROJECT_URLS,
    description="Model mixing using Bayesian Additive Regression Trees",
    long_description=readme_md(),
    long_description_content_type="text/markdown",
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)
