import os
import shutil
import codecs
import platform

from pathlib import Path
from setuptools import setup

# ----- HARDCODED VALUES
# Important paths for our installation structure
PKG_ROOT = Path(__file__).resolve().parent
CLONE_ROOT = PKG_ROOT.parent
SRC_PATH = CLONE_ROOT.joinpath("src")
LIBS_PATH = SRC_PATH.joinpath(".libs")
PKG_SRC_PATH = PKG_ROOT.joinpath("src", "openbtmixing")
BIN_INSTALL_PATH = PKG_SRC_PATH.joinpath(".bin")

# Names of OpenBT command line tools (CLT)
CLT_NAMES = [
    "openbtcli",
    "openbtmixing", "openbtmixingpred", "openbtmixingwts",
    "openbtmopareto",
    "openbtpred",
    "openbtsobol",
    "openbtvartivity"
]

PYTHON_REQUIRES = ">=3.8"
CODE_REQUIRES = ["numpy", "matplotlib"]
TEST_REQUIRES = ["pytest", "scipy", "pandas"]
INSTALL_REQUIRES = CODE_REQUIRES + TEST_REQUIRES

PACKAGE_DATA = {
    'openbtmixing':
        [f".bin/{clt}" for clt in CLT_NAMES] +
        ["tests/bart_bmm_test_data/2d_*.txt"]
}

PROJECT_URLS = {
    "Source": "https://github.com/jcyannotty/OpenBT",
    "Documentation": "https://github.com/jcyannotty/OpenBT",
    "Tracker": "https://github.com/jcyannotty/OpenBT/issues",
}

# ----- FIX BINARY TYPE
# In terms of Python, this package is a pure Python distribution.  However, it
# contains the C++ CLTs/libraries, which have been built for a particular
# system.  Therefore, we alter the name to reflect the correct limitations.
bdist_wheel = {}
system = platform.system().lower()
if system.startswith("darwin"):
    ver, _, arch = platform.mac_ver()
    assert arch in ["x86_64", "arm64"]
    ver = ver.split('.')
    major = int(ver[0])
    minor = int(ver[1])
    if major >= 11:
        # TODO: Ideally we will create wheels with version 10_9 in the name.  I
        # need this to get macOS 14.4 to work in GitHub and it looks like
        # Python is setting mine to 12_0 despite my OS being 12.3.  Is this
        # correct?
        minor = 0
    bdist_wheel["plat_name"] = f"macosx_{major}_{minor}_{arch}"
elif system.startswith("linux"):
    # TODO: No idea what to do here for now.
    arch = platform.machine()
    bdist_wheel["plat_name"] = f"manylinux2014_{arch}"
else:
    raise NotImplementedError(f"Cannot create wheels for {system} yet")

# ----- START CLEAN WITH EVERY DISTRIBUTION
if BIN_INSTALL_PATH.exists():
    assert BIN_INSTALL_PATH.is_dir()
    shutil.rmtree(BIN_INSTALL_PATH)
os.mkdir(BIN_INSTALL_PATH)

# ----- COPY IN OpenBT C++ COMMAND LINE TOOLS
# not the temporary libtool-generated wrapper scripts
for name in CLT_NAMES:
    shutil.copy(str(LIBS_PATH.joinpath(name)), str(BIN_INSTALL_PATH))


# ----- SPECIFY THE PACKAGE
def readme_md():
    fname = PKG_ROOT.joinpath("README.md")
    with codecs.open(fname, "r", encoding="utf8") as fptr:
        return fptr.read()


def version():
    fname = PKG_ROOT.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()


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
    zip_safe=False,
    options={'bdist_wheel': bdist_wheel},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)
