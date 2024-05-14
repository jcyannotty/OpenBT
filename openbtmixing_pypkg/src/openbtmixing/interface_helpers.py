'''
Interface helpers
'''

import numpy as np
import sys
import subprocess
import tempfile
import shutil
import os
import typing

from scipy.stats import norm
from pathlib import Path
from scipy.stats import spearmanr

def read_in_preds(fpath, modelname, nummodels, q_upper, q_lower, readmean, readstd, readwts):
    # Init containers
    mdraws = []
    sdraws = []
    wdraws = {}
    out = {}
    print()
    print(fpath)
    print(modelname)
    print()
    if readmean:
        mdraw_files = sorted(list(fpath.glob(modelname + ".mdraws*")))
        print()
        print(mdraw_files)
        print()
        
        for f in mdraw_files:
            read = open(f, "r")
            lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n':  
                mdraws.append(np.loadtxt(f))

        # print(mdraws[0].shape); print(len(mdraws))
        mdraws = np.concatenate(mdraws, axis=1)  # Got rid of the transpose
        n_test = len(mdraws[0])
        
        pred_mean = np.empty(n_test)
        pred_sd = np.empty(n_test)
        pred_med = np.empty(n_test)
        pred_lower = np.empty(n_test)
        pred_upper = np.empty(n_test)
    
    if readstd:
        sdraw_files = sorted(list(fpath.glob(modelname + ".sdraws*")))
    
        for f in sdraw_files:
            read = open(f, "r")
            lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n':  # If it's nonempty
                sdraws.append(np.loadtxt(f))
        # print(sdraws[0]); print(sdraws[0][0])
        # print(len(sdraws)); print(len(sdraws[0])); print(len(sdraws[0][0]))
        sdraws = np.concatenate(sdraws, axis=1)  # Got rid of the transpose

        n_test = len(sdraws[0])
        sigma_mean = np.empty(n_test)
        sigma_sd = np.empty(n_test)
        sigma_med = np.empty(n_test)
        sigma_lower = np.empty(n_test)
        sigma_upper = np.empty(n_test)

    # Initialize the wdraws dictionary
    if readwts:
        # Get the weight files
        for k in range(nummodels):
            wdraw_files = sorted(list(fpath.glob(modelname + ".w" + str(k + 1) + "draws*")))
            wdraws_temp = []
            for f in wdraw_files:
                read = open(f, "r")
                lines = read.readlines()
                if lines[0] != '\n' and lines[1] != '\n':  # If it's nonempty
                    wdraws_temp.append(np.loadtxt(f))

            # Store the wdraws array in the self.wdraws dictionary under the
            # key wtname
            wtname = "w" + str(k + 1)
            wdraws[wtname] = np.concatenate(wdraws_temp, axis=1) 

        # Initialize summary statistic matrices for the wts
        n_test = len(wdraws[wtname][0])
        wts_mean = np.empty((n_test, nummodels))
        wts_sd = np.empty((n_test, nummodels))
        wts_med = np.empty((n_test, nummodels))
        wts_lower = np.empty((n_test, nummodels))
        wts_upper = np.empty((n_test, nummodels))


    for j in range(n_test):
        if readmean:
            pred_mean[j] = np.mean(mdraws[:, j])
            pred_sd[j] = np.std(mdraws[:, j], ddof=1)
            pred_med[j] = np.quantile(mdraws[:, j], 0.50)
            pred_lower[j] = np.quantile(mdraws[:, j], q_lower)
            pred_upper[j] = np.quantile(mdraws[:, j], q_upper)

        if readstd:
            sigma_mean[j] = np.mean(sdraws[:, j])
            sigma_sd[j] = np.std(sdraws[:, j], ddof=1)
            sigma_med[j] = np.quantile(sdraws[:, j], 0.50)
            sigma_lower[j] = np.quantile(sdraws[:, j], q_lower)
            sigma_upper[j] = np.quantile(sdraws[:, j], q_upper)

        if readwts:
            for k in range(nummodels):
                wtname = "w" + str(k + 1)
                wts_mean[j][k] = np.mean(wdraws[wtname][:, j])
                wts_sd[j][k] = np.std(wdraws[wtname][:, j], ddof=1)
                wts_med[j][k] = np.quantile(wdraws[wtname][:, j], 0.50)
                wts_lower[j][k] = np.quantile(wdraws[wtname][:, j],q_lower)
                wts_upper[j][k] = np.quantile(wdraws[wtname][:, j], q_upper)
            
    if readmean:
        out_pred = {"draws":mdraws,"mean":pred_mean,"sd":pred_sd,
                    "med":pred_med,"lb":pred_lower,"ub":pred_upper}
        out["pred"] = out_pred

    if readstd:
        out_sigma = {"draws":sdraws,"mean":sigma_mean,"sd":sigma_sd,
                "med":sigma_med,"lb":sigma_lower,"ub":sigma_upper}
        out["sigma"] = out_sigma

    if readwts:
        out_wts = {"draws":wdraws,"mean":wts_mean,"sd":wts_sd,
                        "med":wts_med,"lb":wts_lower,"ub":wts_upper}
        out["wts"] = out_wts

    return out


def run_model(fpath, tc, cmd="openbtcli", local_openbt_path = "", google_colab = False):
    """
    Private function, run the cpp program via the command line using
    a subprocess.

    This assumes that mpirun is in the path and that it is from the MPI
    implementation compatible with the build of the OpenBT C++ command line
    tools.

    .. todo::
        * Error check all arguments
        * Error check subprocess call
    """
    if local_openbt_path != "":
        print(f"local_openbt_path = {local_openbt_path}")
        raise NotImplementedError("local_openbt_path not supported")
    if google_colab:
        raise NotImplementedError("google_colab not supported")

    if shutil.which("mpirun") is None:
        msg = "Add to PATH the folder that contains the mpirun\n"
        msg += "of the MPI implementation used to build OpenBT CLTs"
        raise RuntimeError(msg)
    elif shutil.which(cmd) is None:
        raise RuntimeError(f"Add to PATH the folder that contains {cmd}")

    print()
    print(shutil.which("mpirun"))
    print(shutil.which(cmd))
    print()

    # MPI with local program
    try:
        subprocess.run(["mpirun", "-np", str(tc), cmd, str(fpath)],
                       stdin=subprocess.DEVNULL,
                       capture_output=True, check=True)
        #subprocess.run(["mpirun", "-np", str(tc), "--oversubscribe", cmd, str(fpath)],
        #               stdin=subprocess.DEVNULL,
        #               capture_output=True, check=True)
    except subprocess.CalledProcessError as err:
        stdout = err.stdout.decode()
        stderr = err.stderr.decode()
        print()
        msg = "[openbtmixing.mpirun] Unable to run command (Return code {})"
        print(msg.format(err.returncode))
        print("[openbtmixing.mpirun] " + " ".join(err.cmd))
        if stdout != "":
            print("[openbtmixing.mpirun] stdout")
            for line in stdout.split("\n"):
                print(f"\t{line}")
        if stderr != "":
            print("[openbtmixing.mpirun] stderr")
            for line in stderr.split("\n"):
                print(f"\t{line}")
        raise

def write_data_to_txt(data, tc, fpath, root, fmt):
    """
    Private function, write data to text file.
    """
    # Thread count checker
    if (tc - int(tc) != 0):
        sys.exit('Fit: Invalid tc input - exiting process')

    # Writing data for predictions (if true) or training (if false)
    if root in ["xp", "fp", "xw"]:
        nh = tc
        int_added = 0
    else:
        nh = tc - 1
        int_added = 1
    
    # Check splitted data
    splitted_data = np.array_split(data, nh)
    
    # save to text (accounting for thread number when needed with int_added)
    # Use int_added = 0 when writing prediction grid
    for i, ch in enumerate(splitted_data):
        np.savetxt(str(fpath / Path(root + str(i + int_added))),ch, fmt=fmt)



# Need to generalize -- this is only used in fit
def write_config_file(run_params, fpath, tag):
    """
    UPDATE
    """
    configfile = Path(fpath / ("config"+tag) )
    with configfile.open("w") as tfile:
        for param in run_params:
            tfile.write(str(param) + "\n")


def write_train_data(fpath, xi, x_train, y_train, f_train = None, s_train = None, tc = 2):
    """
    Private function, write data to textfiles.
    """
    # print("splits =", splits)
    n = y_train.shape[0]
    write_data_to_txt(y_train, tc,fpath,"y", '%.7f')
    write_data_to_txt(np.transpose(x_train), tc, fpath, "x", '%.7f')
    write_data_to_txt(np.ones(n, dtype="int"),tc, fpath, "s", '%.0f')
    #print("Results stored in temporary path: " + str(self.fpath))
    if x_train.shape[0] == 1:
        np.savetxt(str(fpath / Path("chgv")), [1], fmt='%.7f')
    elif x_train.shape[0] == 2:
        np.savetxt(str(fpath / Path("chgv")),[spearmanr(x_train, axis=1)[0]], fmt='%.7f')
    else:
        np.savetxt(str(fpath / Path("chgv")),spearmanr(x_train, axis=1)[0], fmt='%.7f')

    for k, v in xi.items():
        np.savetxt(str(fpath / Path("xi" + str(k + 1))), v, fmt='%.7f')

    # Write model mixing files
    if not (f_train is None):
        # F-hat matrix
        write_data_to_txt(f_train, tc, fpath, "f", '%.7f')
        # S-hat matrix when using inform_prior
        if not (s_train is None):
            write_data_to_txt(s_train, tc, fpath, "fsd", '%.7f')
        # Wts prior when passed in
        #if self.diffwtsprior:
        #    np.savetxt(str(self.fpath / Path(self.wproot)),
        #                np.concatenate(self.betavec, self.tauvec), fmt='%.7f')


