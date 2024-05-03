import pytest


def test(verbose=0):
    """
    Run the full openbtmixing test suite.

    :param verbose: Verbosity level with 0 being the least verbose; 2, the
        most.
    """
    VALID_LEVELS = [0, 1, 2]

    args = []
    if verbose not in VALID_LEVELS:
        raise ValueError(f"verbose must be in {VALID_LEVELS}")
    elif verbose > 0:
        args = ["-" + "v"*verbose]

    args += ["--pyargs", "openbtmixing.tests"]
    exit_code = pytest.main(args)

    return (exit_code == 0)
