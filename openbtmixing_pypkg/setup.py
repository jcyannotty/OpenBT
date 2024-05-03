import codecs

from pathlib import Path
from setuptools import setup

# ----- HARDCODED VALUES
PKG_ROOT = Path(__file__).resolve().parent

PYTHON_REQUIRES = ">=3.8"
CODE_REQUIRES = ["numpy", "matplotlib"]
TEST_REQUIRES = ["pytest", "scipy", "pandas"]
INSTALL_REQUIRES = CODE_REQUIRES + TEST_REQUIRES

PACKAGE_DATA = {
    'openbtmixing':
        ["tests/bart_bmm_test_data/2d_*.txt"]
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
