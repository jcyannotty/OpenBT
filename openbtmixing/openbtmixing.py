from logging import raiseExceptions
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
import tempfile
import shutil
import os
import typing

from scipy.stats import norm
from pathlib import Path
from scipy.stats import spearmanr

class Openbtmmix():
    # ----------------------------------------------------------
    # "Private" Functions
    def _define_params(self):
        """
        Private function. Set up parameters for the cpp.
        """
        # Overwrite the hyperparameter settings when model mixing
        if self.inform_prior:
            self.tau = 1 / (2 * self.ntree * self.k)
            self.beta = 1 / self.ntree
        else:
            self.tau = 1 / (2 * np.sqrt(self.ntree) * self.k)
            self.beta = 1 / (2 * self.ntree)

        # overall lambda calibration
        if self.overallsd is not None:
            self.overalllambda = self.overallsd**2
        else:
            self.overalllambda = None

        # Set overall nu
        if self.overallnu is None:
            self.overallnu = 10

    def _read_in_preds(self):
        """
        Private function, read in predictions from text files.
        """
        mdraw_files = sorted(
            list(
                self.fpath.glob(
                    self.modelname +
                    ".mdraws*")))
        sdraw_files = sorted(
            list(
                self.fpath.glob(
                    self.modelname +
                    ".sdraws*")))
        mdraws = []
        for f in mdraw_files:
            read = open(f, "r")
            lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n':  # If it's nonempty
                mdraws.append(np.loadtxt(f))
        # print(mdraws[0].shape); print(len(mdraws))
        self.mdraws = np.concatenate(
            mdraws, axis=1)  # Got rid of the transpose
        sdraws = []
        for f in sdraw_files:
            read = open(f, "r")
            lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n':  # If it's nonempty
                sdraws.append(np.loadtxt(f))
        # print(sdraws[0]); print(sdraws[0][0])
        # print(len(sdraws)); print(len(sdraws[0])); print(len(sdraws[0][0]))
        self.sdraws = np.concatenate(
            sdraws, axis=1)  # Got rid of the transpose

        # New (added by me), since R returns arrays like these by default:
        # Calculate mmean and smean arrays, and related statistics
        self.pred_mean = np.empty(len(self.mdraws[0]))
        self.sigma_mean = np.empty(len(self.sdraws[0]))
        self.pred_sd = np.empty(len(self.mdraws[0]))
        self.sigma_sd = np.empty(len(self.mdraws[0]))
        self.pred_5 = np.empty(len(self.mdraws[0]))
        self.sigma_5 = np.empty(len(self.mdraws[0]))
        self.pred_lower = np.empty(len(self.mdraws[0]))
        self.sigma_lower = np.empty(len(self.sdraws[0]))
        self.pred_upper = np.empty(len(self.mdraws[0]))
        self.sigma_upper = np.empty(len(self.sdraws[0]))
        for j in range(len(self.mdraws[0])):
            self.pred_mean[j] = np.mean(self.mdraws[:, j])
            self.sigma_mean[j] = np.mean(self.sdraws[:, j])
            self.pred_sd[j] = np.std(self.mdraws[:, j], ddof=1)
            self.sigma_sd[j] = np.std(self.sdraws[:, j], ddof=1)
            self.pred_5[j] = np.quantile(self.mdraws[:, j], 0.50)
            self.sigma_5[j] = np.quantile(self.sdraws[:, j], 0.50)
            self.pred_lower[j] = np.quantile(self.mdraws[:, j], self.q_lower)
            self.sigma_lower[j] = np.quantile(self.sdraws[:, j], self.q_lower)
            self.pred_upper[j] = np.quantile(self.mdraws[:, j], self.q_upper)
            self.sigma_upper[j] = np.quantile(self.sdraws[:, j], self.q_upper)

    def _read_in_wts(self):
        """
        Private function, read in weights from text files.
        """
        # Initialize the wdraws dictionary
        self.wdraws = {}

        # Initialize summary statistic matrices for the wts
        self.wts_mean = np.empty((self.n_test, self.nummodels))
        self.wts_sd = np.empty((self.n_test, self.nummodels))
        self.wts_5 = np.empty((self.n_test, self.nummodels))
        self.wts_lower = np.empty((self.n_test, self.nummodels))
        self.wts_upper = np.empty((self.n_test, self.nummodels))

        # Get the weight files
        for k in range(self.nummodels):
            wdraw_files = sorted(list(self.fpath.glob(
                self.modelname + ".w" + str(k + 1) + "draws*")))
            wdraws = []
            for f in wdraw_files:
                read = open(f, "r")
                lines = read.readlines()
                if lines[0] != '\n' and lines[1] != '\n':  # If it's nonempty
                    wdraws.append(np.loadtxt(f))

            # Store the wdraws array in the self.wdraws dictionary under the
            # key wtname
            wtname = "w" + str(k + 1)
            self.wdraws[wtname] = np.concatenate(
                wdraws, axis=1)  # Got rid of the transpose

            for j in range(len(self.wdraws[wtname][0])):
                self.wts_mean[j][k] = np.mean(self.wdraws[wtname][:, j])
                self.wts_sd[j][k] = np.std(self.wdraws[wtname][:, j], ddof=1)
                self.wts_5[j][k] = np.quantile(self.wdraws[wtname][:, j], 0.50)
                self.wts_lower[j][k] = np.quantile(
                    self.wdraws[wtname][:, j], self.q_lower)
                self.wts_upper[j][k] = np.quantile(
                    self.wdraws[wtname][:, j], self.q_upper)

    def _run_model(self, cmd="openbtcli"):
        """
        Private function, run the cpp program via the command line using
        a subprocess.
        """
        # Check to see if executable is installed via debian
        sh = shutil.which(cmd)

        # Check to see if installed via wheel
        pyinstall = False
        if sh is None:
            pywhl_path = os.popen("pip show openbtmixing").read()
            pywhl_path = pywhl_path.split("Location: ")
            if len(pywhl_path)>1:
                pywhl_path = pywhl_path[1].split("\n")[0] + "/openbtmixing"
                sh = shutil.which(cmd, path=pywhl_path)
                pyinstall = True

        # Execute the subprocess, changing directory when needed
        if sh is None:
            # openbt exe were not found in the current directory -- try the
            # local directory passed in
            sh = shutil.which(cmd, path=self.local_openbt_path)
            if sh is None:
                raise FileNotFoundError(
                    "Cannot find openbt executables. Please specify the path using the argument local_openbt_path in the constructor.")
            else:
                cmd = sh
                if not self.google_colab:
                    # MPI with local program
                    sp = subprocess.run(["mpirun",
                                         "-np",
                                         str(self.tc),
                                         cmd,
                                         str(self.fpath)],
                                        stdin=subprocess.DEVNULL,
                                        capture_output=True)
                else:
                    # Shell command for MPI with google colab
                    full_cmd = "mpirun --allow-run-as-root --oversubscribe -np " + \
                        str(self.tc) + " " + cmd + " " + str(self.fpath)
                    os.system(full_cmd)
        else:
            if pyinstall:
                # MPI with local program
                #os.system("ldd /home/johnyannotty/Documents/Taweret/test_env/lib/python3.8/site-packages/openbtmixing/.libs/openbtcli")
                #os.environ['LD_LIBRARY_PATH'] = "/home/johnyannotty/Documents/Taweret/test_env/lib/python3.8/site-packages/openbtmixing/.libs/"
                libdir = "/".join(sh.split("/")[:-1]) + "/.libs/"
                os.environ['LD_LIBRARY_PATH'] = libdir
                os.environ['DYLD_LIBRARY_PATH'] = libdir
                cmd = sh
                print(libdir)
                print(cmd)
                sp = subprocess.run(["mpirun",
                                        "-np",
                                        str(self.tc),
                                        cmd,
                                        str(self.fpath)],
                                    stdin=subprocess.DEVNULL,
                                    capture_output=True)
                print(sp)
            else:
                if not self.google_colab:
                    # MPI with installed .exe
                    sp = subprocess.run(["mpirun",
                                        "-np",
                                        str(self.tc),
                                        cmd,
                                        str(self.fpath)],
                                        stdin=subprocess.DEVNULL,
                                        capture_output=True)
                else:
                    # Google colab with installed program
                    full_cmd = "mpirun --allow-run-as-root --oversubscribe -np " + \
                        str(self.tc) + " " + cmd + " " + str(self.fpath)
                    os.system(full_cmd)


    def _set_mcmc_info(self, mcmc_dict):
        """
        Private function, set mcmc information.
        """
        # Extract arguments
        valid_mcmc_args = [
            "ndpost",
            "nskip",
            "nadapt",
            "tc",
            "pbd",
            "pb",
            "stepwpert",
            "probchv",
            "minnumbot",
            "printevery",
            "numcut",
            "adaptevery",
            "xicuts"]
        for (key, value) in mcmc_dict.items():
            if key in valid_mcmc_args:
                self.__dict__.update({key: value})

        # Cutpoints
        if "xicuts" not in self.__dict__:
            self.xi = {}
            maxx = np.ceil(np.max(self.X_train, axis=1))
            minx = np.floor(np.min(self.X_train, axis=1))
            for feat in range(self.p):
                xinc = (maxx[feat] - minx[feat]) / (self.numcut + 1)
                self.xi[feat] = [
                    np.arange(1, (self.numcut) + 1) * xinc + minx[feat]]

        # Birth and Death probability -- set product tree pbd to 0 for selected
        # models
        if (isinstance(self.pbd, float)):
            self.pbd = [self.pbd, 0]

        # Update the product tree arguments
        [self._update_h_args(arg) for arg in ["pbd", "pb", "stepwpert",
                                              "probchv", "minnumbot"]]


    def _set_wts_prior(self, betavec, tauvec):
        """
        Private function, set the non-informative weights prior when
        the weights are not assumed to be identically distributed.
        """
        # Cast lists to np.arrays when needed
        if isinstance(betavec, list):
            betavec = np.array(betavec)
        if isinstance(tauvec, list):
            tauvec = np.array(tauvec)

        # Check lengths
        if not (len(tauvec) == self.nummodels and len(
                betavec) == self.nummodels):
            raise ValueError(
                "Incorrect vector length for tauvec and/or betavec. Lengths must be equal to the number of models.")

        # Store the hyperparameters passed in
        self.diffwtsprior = True
        self.betavec = betavec
        self.tauvec = tauvec


    def _update_h_args(self, arg):
        """
        Private function, update default arguments for the
        product-of-trees model.
        """
        try:
            self.__dict__[arg + "h"] = self.__dict__[arg][1]
            self.__dict__[arg] = self.__dict__[arg][0]
        except BaseException:
            self.__dict__[arg + "h"] = self.__dict__[arg]


    def _write_chunks(self, data, no_chunks, var, *args):
        """
        Private function, write data to text file.
        """
        if no_chunks == 0:
            print("Writing all data to one 'chunk'")
            no_chunks = 1
        if (self.tc - int(self.tc) == 0):
            splitted_data = np.array_split(data, no_chunks)
        else:
            sys.exit('Fit: Invalid tc input - exiting process')
        int_added = 0 if var in ["xp", "fp", "xw"] else 1

        for i, ch in enumerate(splitted_data):
            np.savetxt(str(self.fpath / Path(self.__dict__[var + "root"] + str(i + int_added))),
                       ch, fmt=args[0])

    # Need to generalize -- this is only used in fit

    def _write_config_file(self):
        """
        Private function, create temp directory to write config and data files.
        """
        f = tempfile.mkdtemp(prefix="openbtmixing_")
        self.fpath = Path(f)
        run_params = [
            self.modeltype,
            self.xroot,
            self.yroot,
            self.fmean_out,
            self.ntree,
            self.ntreeh,
            self.ndpost,
            self.nskip,
            self.nadapt,
            self.adaptevery,
            self.tau,
            self.beta,
            self.overalllambda,
            self.overallnu,
            self.base,
            self.power,
            self.baseh,
            self.powerh,
            self.tc,
            self.sroot,
            self.chgvroot,
            self.froot,
            self.fsdroot,
            self.inform_prior,
            self.wproot,
            self.diffwtsprior,
            self.pbd,
            self.pb,
            self.pbdh,
            self.pbh,
            self.stepwpert,
            self.stepwperth,
            self.probchv,
            self.probchvh,
            self.minnumbot,
            self.minnumboth,
            self.printevery,
            self.xiroot,
            self.modelname,
            self.summarystats]
        # print(run_params)
        self.configfile = Path(self.fpath / "config")
        with self.configfile.open("w") as tfile:
            for param in run_params:
                tfile.write(str(param) + "\n")


    def _write_data(self):
        """
        Private function, write data to textfiles.
        """
        splits = (self.n - 1) // (self.n / (self.tc)
                                  )  # Should = tc - 1 as long as n >= tc
        # print("splits =", splits)
        self._write_chunks(self.y_train, splits, "y", '%.7f')
        self._write_chunks(np.transpose(self.X_train), splits, "x", '%.7f')
        self._write_chunks(np.ones((self.n), dtype="int"),
                           splits, "s", '%.0f')
        print("Results stored in temporary path: " + str(self.fpath))
        if self.X_train.shape[0] == 1:
            # print("1 x variable, so correlation = 1")
            np.savetxt(str(self.fpath / Path(self.chgvroot)), [1], fmt='%.7f')
        elif self.X_train.shape[0] == 2:
            # print("2 x variables")
            np.savetxt(str(self.fpath / Path(self.chgvroot)),
                       [spearmanr(self.X_train, axis=1)[0]], fmt='%.7f')
        else:
            # print("3+ x variables")
            np.savetxt(str(self.fpath / Path(self.chgvroot)),
                       spearmanr(self.X_train, axis=1)[0], fmt='%.7f')

        for k, v in self.xi.items():
            np.savetxt(
                str(self.fpath / Path(self.xiroot + str(k + 1))), v, fmt='%.7f')

        # Write model mixing files
        if self.modeltype == 9:
            # F-hat matrix
            self._write_chunks(self.F_train, splits, "f", '%.7f')
            # S-hat matrix when using inform_prior
            if self.inform_prior:
                self._write_chunks(self.S_train, splits, "fsd", '%.7f')
            # Wts prior when passed in
            if self.diffwtsprior:
                np.savetxt(str(self.fpath / Path(self.wproot)),
                           np.concatenate(self.betavec, self.tauvec), fmt='%.7f')