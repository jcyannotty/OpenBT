from pathlib import Path


# ----- INSTALLATION DETAILS
# Location of MPI binaries for running MPI programs
__MPI_BIN_PATH = Path(__file__).parent.joinpath("mpi", "bin").resolve()
MPIRUN = str(__MPI_BIN_PATH.joinpath("mpirun"))
MPIEXEC = str(__MPI_BIN_PATH.joinpath("mpiexec"))
