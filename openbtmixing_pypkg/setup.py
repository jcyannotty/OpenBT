import os
import codecs

import subprocess as sbp

from pathlib import Path
from setuptools import (
    setup, Command
)
from setuptools.command.build import build as _build

# ----- HARDCODED VALUES
PKG_ROOT = Path(__file__).resolve().parent
PY_SRC_PATH = PKG_ROOT.joinpath("src", "openbtmixing")
CLT_SRC_PATH = PKG_ROOT.joinpath("cpp")

# Names of C++ products to include
#
# Only a subset of OpenBT command line tools are used in package
CLT_NAMES = [
    "openbtcli",
    "openbtpred",
    "openbtmixingwts"
]
LIB_BASENAME = "libopenbtmixing"

# Package metadata
PYTHON_REQUIRES = ">=3.9"
CODE_REQUIRES = ["numpy", "matplotlib"]
TEST_REQUIRES = ["pytest", "scipy", "pandas"]
INSTALL_REQUIRES = CODE_REQUIRES + TEST_REQUIRES

PACKAGE_DATA = {
    "openbtmixing":
        ["tests/bart_bmm_test_data/2d_*.txt"] + \
        [f"bin/{clt}" for clt in CLT_NAMES] + \
        [f"lib/{LIB_BASENAME}.*"]
}

PROJECT_URLS = {
    "Source": "https://github.com/jcyannotty/OpenBT",
    "Documentation": "https://github.com/jcyannotty/OpenBT",
    "Tracker": "https://github.com/jcyannotty/OpenBT/issues",
}

# ----- CUSTOM COMMAND TO BUILD C++ CLTs
# EdgeDB builds a CLT that they make available through their interface.  This
# is partially based on what they have done at commit c5db98c.
#
# https://github.com/edgedb/edgedb/blob/master/setup.py
#
# TODO: Is there a way to get the C++ code into source distributions without
# having to use the symlinks?  I recall seeing an repo that claimed that these
# facilities can do that.
class build(_build):
    user_options = _build.user_options
    sub_commands = ([("build_clt", lambda self: True)])


class build_clt(Command):
    description = "Build the OpenBTMixing CLTs"
    user_options: list[str] = []
    editable_mode: bool

    def initialize_options(self):
        # TODO: Can we get this to work with editable mode installations?
        self.editable_mode = False

    def finalize_options(self):
        pass

    def run(self, *args, **kwargs):
        SETUP_CMD = ["meson", "setup", "--wipe", "--buildtype=release",
                     "builddir",
                     f"-Dprefix={PY_SRC_PATH}",
                     "-Dmpi=open-mpi"]
        COMPILE_CMD = ["meson", "compile", "-C", "builddir"]
        INSTALL_CMD = ["meson", "install", "--quiet", "-C", "builddir"]

        # Install the CLTs within the Python source files and so that they are
        # included in the wheel build based on PACKAGE_DATA
        # TODO: Error check all the calls.
        cwd = Path.cwd()
        os.chdir(CLT_SRC_PATH)
        for cmd in [SETUP_CMD, COMPILE_CMD, INSTALL_CMD]:
            sbp.run(cmd)
        os.chdir(cwd)

cmdclass = {
    'build': build,
    'build_clt': build_clt
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
    cmdclass=cmdclass,
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
