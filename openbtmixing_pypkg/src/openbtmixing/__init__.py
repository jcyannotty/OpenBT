from importlib.metadata import version

from .openbtmixing import Openbtmix

from .test import test

__version__ = version("openbtmixing")

print()
print(
    "WARNING - This package is no longer supported.\n"
    "Consider using its replacment https://github.com/bandframework/OpenBT"
)
print()
