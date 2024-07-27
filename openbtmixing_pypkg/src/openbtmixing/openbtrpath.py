import numpy as np
import tempfile
from pathlib import Path

import openbtmixing_pypkg.interface_helpers as ih
from openbtmixing_pypkg.openbtmixing import Openbtmix


class OpenbtRpath(Openbtmix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modeltype = 10 
        self.nummodels = 1       
    

    def set_prior(
        self,
        ntree: int = 1,
        ntreeh: int = 1,
        k: float = 2,
        power: float = 2.0,
        base: float = 0.95,
        sighat: float = 1,
        nu: int = 10,
        rpath = False,
        a1 = 2,
        a2 = 2,
        q = 4,
        ymin = 0,
        ymax = 1):
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
        :param bool rpath:
            True if using the random path model
        :param float a1:
            The first shape parameter for the Beta prior on the bandwith parameter in the random path model.
        :param float a2:
            The second shape parameter for the Beta prior on the bandwith parameter in the random path model.
        :param float q:
            The shape parameter in the random path splitting probabilities.
        :param float ymin:
            The minimum value of the observed process.
        :param float ymax:
            The maximum value of the observed process.    
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
            'rpath':rpath,
            'q':q,
            'shape1':a1,
            'shape2':a2,
            'gamma':a1/(a1+a2),
            'ymin':ymin,
            'ymax':ymax}

        self._prior = dict()
        for (key, value) in prior_dict.items():
            # if key in valid_prior_args:
            # Update class dictionary which stores all objects
            self.__dict__.update({key: value})
            # Updates prior specific dictionary
            self._prior.update({key: value})

        # Set lambda and nu
        self.nu = nu
        self.lam = ((nu + 2)/nu)*sighat*sighat

        # Set tau and the prior mean
        self.beta = 0
        self.tau = (ymax - ymin)/(2*k*np.sqrt(self.ntree))
    
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs):
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
        
        # Store n and p
        n = x_train.shape[0]
        self.p = x_train.shape[1]

        # Set the default for f_train
        f_train = np.ones(n).reshape(n,1)

        # Transpose x
        x_train = np.transpose(x_train)

        # Set default outputs
        self.fmean_out = np.mean(y_train)
        y_train = y_train - self.fmean_out
        
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
        modbd = False
        self.batchsize = self.ndpost
        
        # Get config parameters
        run_params = [self.modeltype,self.xroot,self.yroot,self.ntree,self.ntreeh,
                        self.ndpost,self.nskip,self.nadapt,self.adaptevery,self.tau,self.beta,
                        self.lam,self.nu,self.base,self.power,self.baseh,self.powerh,self.maxd,
                        self.tc,self.sroot,self.chgvroot,self.froot,self.fsdroot,self.inform_prior,
                        self.rpath,self.gamma,self.q,self.shape1,self.shape2,modbd,
                        self.pbd,self.pb,self.pbdh,self.pbh,
                        self.stepwpert,self.stepwperth,self.probchv,self.probchvh,self.minnumbot,
                        self.minnumboth,self.printevery,self.batchsize,
                        self.xiroot,self.modelname,self.summarystats]

        # Write config file
        ih.write_config_file(run_params, self.fpath, tag = "")
        ih.write_train_data(self.fpath, self.xi, x_train, y_train, f_train, s_train = None, tc = self.tc)

        print("Running model...")        
        cmd = "openbtmixingts"
        self.sp_train = ih.run_model(self.fpath, self.tc, cmd, local_openbt_path = self.local_openbt_path, google_colab = self.google_colab)

        # See which attributes are returned/can be returned here
        # Return attributes to be saved as a separate fit object:
        # Missing the influence attribute from the R code (skip for now)
        res = {}
        keys = ['fpath']

        for key in keys:
            res[key] = self.__dict__[key]

        return res


    def predict(self, x_test: np.ndarray, ci: float = 0.95):
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
        n_test = x_test.shape[0]

        # Set the default for f_train
        f_test = np.ones(n_test).reshape(n_test,1)


        if p_test != self.p:
            raise ValueError("Number of inputs in x_test does not match number of inputs in x_train.")
        
        if f_test.shape[1] != self.nummodels:
            raise ValueError("Number of models in f_test does not equal number of models in f_train.")

        self.xproot = "xp"
        self.fproot = "fp"
        self.gammaroot = "rpg"

        ih.write_data_to_txt(x_test, self.tc, self.fpath, self.xproot,'%.7f')
        ih.write_data_to_txt(f_test, self.tc, self.fpath, self.fproot,'%.7f')

        # Defaults that are used in the config for the batch mixing cpp file
        temperature = 0.5
        domdraws = True

        dosdraws = True
        dopdraws = False
        dosoftmax = False
        self.numbatches = 1

        # Set and write config file
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xproot, self.fproot,
                       self.ndpost, self.ntree, self.ntreeh,
                       p_test, self.nummodels,temperature,domdraws,
                       dosdraws, dopdraws, dosoftmax, self.batchsize,
                       self.numbatches, self.tc,
                       self.rpath, self.gammaroot]
        
        ih.write_config_file(pred_params, self.fpath, tag = ".pred")

        # Run prediction
        cmd = "openbtmixingpredts"
        self.sp_pred = ih.run_model(fpath = self.fpath, tc = self.tc, 
                     cmd = cmd, local_openbt_path = self.local_openbt_path, 
                     google_colab = self.google_colab)
        
        res = ih.read_in_preds(self.fpath, self.modelname, 
                               self.nummodels, q_upper, q_lower, 
                               domdraws, dosdraws, False, self.fmean_out)        
        return res


    def predict_weights(self, x_test: np.ndarray, ci: float = 0.95):
        raise ValueError("Not used for RPBART univariate regression...")


