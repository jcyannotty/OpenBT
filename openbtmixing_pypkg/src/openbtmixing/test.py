import pytest


def test(level=0):
    """
    Execute openbtmixing's full test suite.

    :param level: Level 0 indicates the least logging.  Level 2 indicates the
        maximum logging possible.
    """
    VERBOSITY = [0, 1, 2]

    args = []
    if level not in VERBOSITY:
        raise ValueError(f"level must be in {VERBOSITY}")
    elif level >= 1:
        args = ["-" + "v"*level]

    return pytest.main(args + ["--pyargs", "openbtmixing.tests"])
