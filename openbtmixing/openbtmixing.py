from logging import raiseExceptions
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
import tempfile
import shutil
import os
import typing


from pathlib import Path

import openbtmixing.interface_helpers as ih

#from importlib import reload
#reload(ih)

class Openbtmix():
    def __init__(self,**kwargs):
        '''

        Parameters:
        ----------
        :param dict model_dict:
            Dictionary of models where each item is an instance of BaseModel.

        :param dict kwargs:
            Additional arguments to pass to the constructor.

        Returns:
        ---------
        :returns: None.
        '''

        # MCMC Parameters
        self.ndpost = 1000
        self.nskip = 100
        self.nadapt = 1000
        self.tc = 2
        self.pbd = 0.7
        self.pb = 0.5
        self.stepwpert = 0.1
        self.probchv = 0.1
        self.minnumbot = 5
        self.printevery = 100
        self.numcut = 100
        self.adaptevery = 100

        # Cutpoints
        self.xicuts = None

        # Set the prior defaults
        self.nummodels = 2
        self.sighat = 1
        self.nu = 10
        self.k = 2
        self.ntree = 1
        self.ntreeh = 1
        self.power = 2.0
        self.base = 0.95
        self.inform_prior = False

        # Set defaults for variance product trees, assume constant variance in this model
        self.baseh = 0.95
        self.powerh = 2.0
        self.minnumboth = 5
        self.pbdh = 0
        self.pbh = 0
        self.stepwperth = 0.1
        self.probchvh = 0

        # Define the roots for the output files --- might be able to delete after hard code
        self.xroot = "x"
        self.yroot = "y"
        self.sroot = "s"
        self.chgvroot = "chgv"
        self.froot = "f"
        self.fsdroot = "fsd"
        self.wproot = "wpr"
        self.xiroot = "xi"

        # Set other defaults
        self.modelname = "mixmodel"
        self.summarystats = "FALSE"
        self.local_openbt_path = os.getcwd()
        self.google_colab = False       
        self.diffwtsprior = False # used for diff values of beta and tau in noninform prior
        self.tauvec = None # the tauvec for diff components in noninform prior
        self.betavec = None # the betavec for diff components in noninform prior

        # Set the kwargs dictionary
        self.__dict__.update((key, value) for key, value in kwargs.items())

        # Set defaults not relevant to model mixing -- only set so cpp code
        # doesn't break
        self.truncateds = None
        self.modeltype = 9
        self._is_prior_set = False
        self._is_predict_run = False  # Remove ????

        # Setter for prior to compute tau, beta, ect
        self.set_prior(self.ntree,self.ntreeh,self.k,self.power,self.base,
                       self.sighat,self.nu,self.inform_prior)


    def get_prior(self):
        out = {}
        out["beta"] = self.beta
        out["tau"] = self.tau
        out["nu"] = self.nu
        out["lam"] = self.lam
        out["base"] = self.base
        out["power"] = self.power
        return out


    def get_mcmc_info(self):
        out = {}
        out["pbd"] = self.pbd
        out["pb"] = self.pb
        out["stepwpert"] = self.stepwpert
        out["probchv"] = self.probchv
        out["nadapt"] = self.nadapt
        out["adaptevery"] = self.adaptevery
        out["nskip"] = self.nskip
        out["ndpost"] = self.ndpost
        return out

    
    
    def set_cutpoints(self, xmax, xmin, numcut):
        self.xi = {}
        self.numcut = numcut
        for j in range(len(xmax)):
            xinc = (xmax[j] - xmin[j]) / (self.numcut + 1)
            self.xi[j] = [np.arange(1, (numcut) + 1) * xinc + xmin[j]]


    def set_mcmc_info(self, mcmc_dict):
        # Extract arguments
        valid_mcmc_args = [
            "ndpost",
            "nskip",
            "nadapt",
            "adaptevery",
            "tc",
            "minnumbot",
            "printevery"]
        
        for (key, value) in mcmc_dict.items():
            if key in valid_mcmc_args:
                self.__dict__.update({key: value})
            else:
                print((key,value))


    def set_prior(
            self,
            ntree: int = 1,
            ntreeh: int = 1,
            k: float = 2,
            power: float = 2.0,
            base: float = 0.95,
            sighat: float = 1,
            nu: int = 10,
            inform_prior: bool = True):
        '''
        Sets the hyperparameters in the tree and terminal node priors. Also
        specifies if an informative or non-informative prior will be used when mixing EFTs.

        Parameters:
        -----------
        :param int ntree:
            The number of trees used in the sum-of-trees model for
            the weights.
        :param int ntreeh:
            The number of trees used in the product-of-trees model
            for the error standard deviation. Set to 1 for
            homoscedastic variance assumption.
        :param float k:
            The tuning parameter in the prior variance of the terminal node
            parameter prior. This is a value greater than zero.
        :param float power:
            The power parameter in the tree prior.
        :param float base:
            The base parameter in the tree prior.
        :param float overallsd:
            An initial estimate of the erorr standard deviation.
            This value is used to calibrate the scale parameter in variance prior.
        :param float overallnu:
            The shape parameter in the error variance prior.
        :param bool inform_prior:
            Controls if the informative or non-informative prior is used.
            Specify true for the informative prior.

        Returns:
        --------
        :returns: None.

        '''
        # Extract arguments
        # valid_prior_args = ['ntree', 'ntreeh', 'k','power','base','overallsd','overallnu','inform_prior','tauvec', 'betavec']
        prior_dict = {
            'ntree': ntree,
            'ntreeh': ntreeh,
            'k': k,
            'power': power,
            'base': base,
            'sighat': sighat,
            'nu': nu,
            'inform_prior': inform_prior}

        self._prior = dict()
        for (key, value) in prior_dict.items():
            # if key in valid_prior_args:
            # Update class dictionary which stores all objects
            self.__dict__.update({key: value})
            # Updates prior specific dictionary
            self._prior.update({key: value})


        # Run _define_parameters
        if self.inform_prior:
            self.tau = 1 / (2*self.ntree*self.k)
            self.beta = 1 / self.ntree
        else:
            self.tau = 1 / (2*np.sqrt(self.ntree)*self.k)
            self.beta = 1 / (self.nummodels*self.ntree)

        # Set lambda and nu
        self.nu = nu
        self.lam = ((nu + 2)/nu)*sighat*sighat
        

    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, f_train: np.ndarray,
              s_train = None, **kwargs):
        '''
        Train the mixed-model using a set of observations y at inputs x.

        Parameters:
        ----------
        :param np.ndarray X: input parameter values of dimension (n x p).
        :param np.ndarray y: observed data at inputs X of dimension  (n x 1).
        :param dict kwargs: dictionary of arguments

        Returns:
        --------
        :returns: A dictionary which contains relevant information to the model such as
            values of tuning parameters. The MCMC results are written to a text file
            and stored in a temporary directory as defined by the fpath key in the
            results dictionary.
        :rtype: dict

        '''
        # Cast data to arrays if not already and reshape if needed
        if isinstance(x_train, list):
            x_train = np.array(x_train)

        if len(x_train.shape) == 1:
            x_train = x_train.reshape(x_train.shape[0], 1)

        if isinstance(y_train, list):
            y_train = np.array(y_train)

        if len(y_train.shape) == 1:
            y_train = y_train.reshape(y_train.shape[0], 1)

        # Dimension Checks
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of rows in x_train does not match length of y_train.")

        if x_train.shape[0] != f_train.shape[0]:
            raise ValueError("Number of rows in x_train does not match length of f_train.")
        
        if self.inform_prior:
            if f_train.shape != s_train.shape:
                raise ValueError("Shape of f_train does not match shape of s_train.")
        
        # Store n and p
        n = x_train.shape[0]
        self.p = x_train.shape[1]

        # Transpose x
        x_train = np.transpose(x_train)

        # Set default outputs
        self.fmean_out = 0
        
        # Cutpoints
        if self.__dict__["xicuts"] is None:
            # Get the bounds
            xmax = np.ceil(np.max(x_train, axis=1))
            xmin = np.floor(np.min(x_train, axis=1))
            if "numcut" in kwargs.keys():
                self.numcut = kwargs["numcut"]    
            self.set_cutpoints(xmax, xmin, self.numcut)


        # Set the mcmc properties from kwargs
        self.set_mcmc_info(kwargs)

        # Create path
        f = tempfile.mkdtemp(prefix="openbtmixing_")
        self.fpath = Path(f)

        # Get config parameters
        run_params = [self.modeltype,self.xroot,self.yroot,self.fmean_out,self.ntree,self.ntreeh,
                        self.ndpost,self.nskip,self.nadapt,self.adaptevery,self.tau,self.beta,
                        self.lam,self.nu,self.base,self.power,self.baseh,self.powerh,self.tc,
                        self.sroot,self.chgvroot,self.froot,self.fsdroot,self.inform_prior,
                        self.wproot,self.diffwtsprior,self.pbd,self.pb,self.pbdh,self.pbh,
                        self.stepwpert,self.stepwperth,self.probchv,self.probchvh,self.minnumbot,
                        self.minnumboth,self.printevery,self.xiroot,self.modelname,self.summarystats]


        # Write config file
        ih.write_config_file(run_params, self.fpath, tag = "")
        if self.inform_prior:
            ih.write_train_data(self.fpath, self.xi, x_train, y_train, f_train, s_train = s_train, tc = self.tc)
        else:
            ih.write_train_data(self.fpath, self.xi, x_train, y_train, f_train, s_train = None, tc = self.tc)

        print("Running model...")        
        cmd = "openbtcli"
        ih.run_model(self.fpath, self.tc, cmd, local_openbt_path = self.local_openbt_path, google_colab = self.google_colab)

        # See which attributes are returned/can be returned here
        # Return attributes to be saved as a separate fit object:
        # Missing the influence attribute from the R code (skip for now)
        res = {}
        keys = ['fpath','inform_prior','minnumbot','nummodels','p']

        for key in keys:
            res[key] = self.__dict__[key]

        return res

    
    def predict(self, x_test: np.ndarray, f_test: np.ndarray, ci: float = 0.95):
        '''
        Obtain the posterior predictive distribution of the mixed-model at a set
        of inputs X.

        Parameters:
        ----------
        :param np.ndarray X: design matrix of testing inputs.
        :param float ci: credible interval width, must be a value within the interval (0,1).

        Returns:
        --------
        :returns: The posterior prediction draws and summaries.
        :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        :return value: the posterior predictive distribution evaluated at the specified test points
        :return value: the posterior mean of the mixed-model at each input in X.
        :return value: the pointwise credible intervals at each input in X.
        :return value: the posterior standard deviation of the mixed-model at each input in X.
        '''

        # Set q_lower and q_upper
        alpha = (1 - ci)
        q_lower = alpha / 2
        q_upper = 1 - alpha / 2

        # Casting lists to arrays when needed
        if (isinstance(x_test, list)):
            x_test = np.array(x_test)
        if (len(x_test.shape) == 1):  # If shape is (n, ), change it to (n, 1):
            x_test = x_test.reshape(len(x_test), 1)


        # Set control values
        p_test = x_test.shape[1]

        if p_test != self.p:
            raise ValueError("Number of inputs in x_test does not match number of inputs in x_train.")
        
        if f_test.shape[1] != self.nummodels:
            raise ValueError("Number of models in f_test does not equal number of models in f_train.")

        self.xproot = "xp"
        self.fproot = "fp"

        ih.write_data_to_txt(x_test, self.tc, self.fpath, self.xproot,'%.7f')
        ih.write_data_to_txt(f_test, self.tc, self.fpath, self.fproot,'%.7f')

        # Set and write config file
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xproot, self.fproot,
                       self.ndpost, self.ntree, self.ntreeh,
                       p_test, self.nummodels, self.tc, self.fmean_out]
        ih.write_config_file(pred_params, self.fpath, tag = ".pred")

        # Run prediction
        cmd = "openbtpred"
        ih.run_model(fpath = self.fpath, tc = self.tc, cmd = cmd, local_openbt_path = self.local_openbt_path, google_colab = self.google_colab)
        
        res = ih.read_in_preds(self.fpath, self.modelname, self.nummodels, q_upper, q_lower, True, True, False)        
        return res

    
    def predict_weights(self, x_test: np.ndarray, ci: float = 0.95):
        '''
        Obtain posterior distribution of the weight functions at a set
        of inputs X.

        Parameters:
        ----------
        :param np.ndarray X: design matrix of testing inputs.
        :param float ci: credible interval width, must be a value within the interval (0,1).

        Returns:
        --------
        :returns: The posterior weight function draws and summaries.
        :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        :return value: the posterior draws of the weight functions at each input in X.
        :return value: posterior mean of the weight functions at each input in X.
        :return value: pointwise credible intervals for the weight functions.
        :return value: posterior standard deviation of the weight functions at each input in X.
        '''

        # Set q_lower and q_upper
        alpha = (1 - ci)
        q_lower = alpha / 2
        q_upper = 1 - alpha / 2

        # Checks for proper inputs and convert lists to arrays
        if not self.modeltype == 9:
            raise TypeError("Cannot call openbt.mixingwts() method for openbt objects that are not modeltype = 'mixbart'")
        if isinstance(x_test, list):
            x_test = np.array(x_test)
        if (len(x_test.shape) == 1):
            x_test = x_test.reshape(len(x_test), 1)

        # Set control values
        p_test = x_test.shape[1]
        if p_test != self.p:
            raise ValueError("Number of inputs in x_test does not match number of inputs in x_train.")


        # Set control parameters
        self.xwroot = "xw"
        # default, needed when considering the general class of model mixing
        # problems -- revist this later
        self.fitroot = ".fit"

        # write out the config file
        # Set control values
        # no need to set this as a class object like in predict function
        ih.write_data_to_txt(x_test, self.tc, self.fpath, self.xwroot,'%.7f')

        # Set and write config file
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xwroot, self.fitroot,
                       self.ndpost, self.ntree, self.ntreeh,
                       p_test, self.nummodels, self.tc]
        
        ih.write_config_file(pred_params, self.fpath, tag = ".mxwts")
        
        # run the program
        cmd = "openbtmixingwts"
        ih.run_model(self.fpath, self.tc, cmd, local_openbt_path = self.local_openbt_path, google_colab = self.google_colab)

        res = ih.read_in_preds(self.fpath, self.modelname, self.nummodels, q_upper, q_lower, False, False, True)        
        return res

