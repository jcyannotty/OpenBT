import pandas as pd
import numpy as np
import math
import scipy
import gsum as gm
import matplotlib.pyplot as plt

from itertools import chain
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#------------------------------------------------------
# Define Small-g expansion terms -- using coefs and Q(x)
def fsg_terms(x,k):
    #Get the coefficients -- only even coefficients are non-zero
    # Get q(x) and yref(x)
    if k % 2 == 0:
        sk = np.sqrt(2)*scipy.special.gamma(k + 1/2)*(-4)**(k/2)/math.factorial(k/2)
        qx = x
        yref = x*math.factorial(k/2 + 2)/np.sqrt(1-x**2)
    else:
        sk = 0
        qx = x
        yref = 0
    # Get the polynomial term
    tx = sk*x**k
    out = {'tx':tx,'cx':sk,'qx':qx,'yref':yref}
    return out

# Define Large-g expansion terms -- using coefs and Q(x)
def flg_terms(x,k):
    #Get the coefficients
    lk = scipy.special.gamma(k/2 + 0.25)*(-0.5)**(k)/(2*math.factorial(k))
  
    #Get the expansion coef, q, and term
    tx = lk*x**(-k)/np.sqrt(x)
    yref = (1/(math.factorial(k+1)*x**(k+1.5)))*(0.75/0.5**(2*k+2))
    qx = 0.5
    
    out = {'tx':tx, 'cx':lk, 'qx':qx, 'yref':yref}
    return out


# Get polynomial terms for the model at the specified inputs
def get_terms(x_list,k, model = 'sg'):
    term_list = []
    coef_list = []
    qx_list = []
    yref_list = []
    for x in x_list:
        if model == 'sg':
            res = fsg_terms(x,k)
        elif model == 'lg':
            res = flg_terms(x,k)
        else:
            raise ValueError('Wrong model specified. Valid arguments are either sg or lg.') 
        tx = res['tx']
        cx = res['cx'] 
        qx = res['qx']
        yref = res['yref']
        term_list.append(tx)
        coef_list.append(cx)
        qx_list.append(qx)
        yref_list.append(yref)
      
    out = {'term_list':term_list, 'coef_list':coef_list, 'qx_list':qx_list, 'yref_list':yref_list}
    return out

# Define the expansions
def get_exp(x_list,k, model = 'sg'):
    cols = []
    for i in list(range(0,k+1)):
        cols.append('order'+str(i))
    coef_df = pd.DataFrame(columns = cols)
    term_df = pd.DataFrame(columns = cols)
    ypred_df = pd.DataFrame(columns = cols)
    qx_df = pd.DataFrame(columns = ['qx'])
    yref_df = pd.DataFrame(columns = ['yref'])
    for i in range(0,k+1):
        res = get_terms(x_list, i, model)
        term_df.iloc[:,i] = res['term_list']
        coef_df.iloc[:,i] = res['coef_list']
        if i == k:
            qx_df.iloc[:,0] = res['qx_list']
            yref_df.iloc[:,0] = res['yref_list']
        if i >= 1:
            ypred_df.iloc[:,i] = term_df.iloc[:,i] + ypred_df.iloc[:,i-1]
        else:
            ypred_df.iloc[:,i] = term_df.iloc[:,i]
    out = {'coef_df':coef_df, 'term_df':term_df, 'ypred_df':ypred_df, 'qx_df':qx_df,'yref_df':yref_df}
    return out 

# Checker
fsg_terms(0.03, 2)
flg_terms(0.1, 2)

# Set training points for model runs
#x1_train = np.linspace(0.05, 0.5, 5)
#x2_train = np.linspace(0.05, 0.5, 5)

x1_train = np.array([0.03, 0.10, 0.15, 0.25])
x2_train = np.array([0.25, 0.35,0.40,0.50])

x1_test = np.linspace(0.03, 0.5, 20)
x2_test = np.linspace(0.03, 0.5, 20)

#x1_test = np.linspace(0.03, 0.5, 300)
#x2_test = np.linspace(0.03, 0.5, 300)

x1_data = np.concatenate((x1_train,x1_test), axis = 0)
x2_data = np.concatenate((x2_train,x2_test), axis = 0)

#Get coefficients from both models
ns = 2
c1_cols = []
c1_list = []
for i in range(ns+1): 
    c1_cols.append('order' + str(i))
    c1_data = get_terms(x1_train, i,'sg')['coef_list']
    c1_list.append(c1_data)
c1_array = np.array(c1_list).transpose()

nl = 4
c2_cols = []
c2_list = []
for i in range(nl+1): 
    c2_cols.append('order' + str(i))
    c2_data = get_terms(x2_train, i,'lg')['coef_list']
    c2_list.append(c2_data)
c2_array = np.array(c2_list).transpose()

# Convert the x's to required format
x1_train = x1_train[:,None]
x2_train = x2_train[:,None]

x1_test = x1_test[:,None]
x2_test = x2_test[:,None]

x1_data = x1_data[:,None]
x2_data = x2_data[:,None]

#-------------------------
# Hyperparameters
center0 = 0
disp0 = 0
df0 = 100
scale0 = 2.5
kernel1 = RBF(length_scale=0.3, length_scale_bounds=(0.05, 4)) + WhiteKernel(1e-10, noise_level_bounds='fixed')
kernel2 = RBF(length_scale=0.5, length_scale_bounds=(0.05, 1)) + WhiteKernel(1e-10, noise_level_bounds='fixed')

exp1_res = get_exp(list(chain(*x1_data.tolist())), 2, 'sg')
y1_df = exp1_res['ypred_df']
coeffs1 = gm.coefficients(np.array(y1_df),ratio = 1,orders=np.arange(0,ns+1))

gp1 = gm.ConjugateGaussianProcess(kernel1, center=center0, disp=disp0, df=df0, scale=5, n_restarts_optimizer=10)
fit1 = gp1.fit(x1_train, coeffs1[:4,])
fit1.predict(x1_test,return_std=True)
np.sum(fit1.predict(x1_test,return_std=False),axis = 1)
fit1.cbar_sq_mean_

ratio1 = np.array(list(chain(*exp1_res['qx_df'].values.tolist()))) # Reshape the array
ref1 = np.array(list(chain(*exp1_res['yref_df'].values.tolist()))) # Reshape the array

gp_trunc1 = gm.TruncationGP(kernel = fit1.kernel_, center=center0, disp=disp0, df=df0, scale=scale0, ref = ref1, ratio = ratio1)
fit_trunc1 = gp_trunc1.fit(x1_data, y=np.array(exp1_res['ypred_df']), orders=np.arange(0,ns+1))
fit_trunc1.predict(x1_data, order = 2, return_std = True)

fhat1 = fit_trunc1.predict(x1_data, order = 2, return_std = False)
std1 = fit_trunc1.predict(x1_data, order = 2, return_std = True)[1]

fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.plot(x1_test, fhat1[5:], zorder = 2)
#ax.fill_between(x1_test[:,0], fhat1[5:] + 2*std1[5:], fhat1[5:] - 2*std1[5:], zorder=1)
plt.show()


center0 = 0
disp0 = 0
df0 = 10
scale0 = 0.2

exp2_res = get_exp(list(chain(*x2_data.tolist())), 4, 'lg')
y2_df = exp2_res['ypred_df']

ratio2 = np.array(list(chain(*exp2_res['qx_df'].values.tolist()))) # Reshape the array
ref2 = np.array(list(chain(*exp2_res['yref_df'].values.tolist()))) # Reshape the array

coeffs2 = gm.coefficients(np.array(y2_df),ratio = ref2,orders=np.arange(0,nl+1))

gp2 = gm.ConjugateGaussianProcess(kernel2, center=center0, disp=disp0, df=df0, scale=5, n_restarts_optimizer=10)
fit2 = gp2.fit(x2_train, coeffs2[:4,])
#fit2.predict(x2_test,return_std=True)
#np.sum(fit2.predict(x2_test,return_std=False),axis = 1)
fit2.cbar_sq_mean_

gp_trunc2 = gm.TruncationGP(kernel = fit2.kernel_, center=center0, disp=disp0, df=df0, scale=scale0, ref = ref2, ratio = ratio2)
fit_trunc2 = gp_trunc2.fit(x2_data, y=np.array(exp2_res['ypred_df']), orders=np.arange(0,nl+1))
#fit_trunc2.predict(x2_data, order = 4, return_std = True)

fhat2 = fit_trunc2.predict(x2_data, order = 4, return_std = False)
std2 = fit_trunc2.predict(x2_data, order = 4, return_std = True)[1]

fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.plot(x2_test, fhat2[5:], zorder = 2)
ax.fill_between(x1_test[:,0], fhat2[5:] + 2*std2[5:], fhat2[5:] - 2*std2[5:], zorder=1)
plt.show()

#-----------------------------------------------------
nn = fhat1.shape[0]
fdata = np.concatenate([fhat1.reshape(nn,1), fhat2.reshape(nn,1)], axis = 1)
sdata = np.concatenate([std1.reshape(nn,1), std2.reshape(nn,1)], axis = 1)

np.savetxt("/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Overleaf Results/f_train.txt", fdata[4:])
np.savetxt("/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Overleaf Results/s_train.txt", sdata[4:])


#-----------------------------------------------------
# Model Mixing
n_train = 20
n_test = 100
s = 0.03

x_train = x1_data[4:]
x_grid = x1_test
#x_test = np.linspace(0.01, 1.0, n_test)

np.random.seed(1234567)
y_train = np.array([2.508488,2.481616,2.462142,2.436121,2.405677,2.374887,2.343086,2.305914,2.271658,2.232709,2.203263,2.175726,2.136858,
    2.116650, 2.068122, 2.040274, 2.017935, 1.990049, 1.964042, 1.937480])
nn = fhat1.shape[0]
f_train = np.concatenate([fhat1.reshape(nn,1), fhat2.reshape(nn,1)], axis = 1)
s_train = np.concatenate([std1.reshape(nn,1), std2.reshape(nn,1)], axis = 1)

from openbt import OPENBT
m = OPENBT(model = "mixbart", tc = 4, modelname = "eft", ntree = 10, k = 1, ndpost = 10000, nskip = 2000, nadapt = 5000, 
                adaptevery = 500, overallsd = 0.01, minnumbot = 3, nsprior = True, overallnu = 5, numcut = 300)
fit = m.fit(x_train, y_train, F= f_train[4:(n_train+4),:], S= s_train[4:(n_train+4),:])
f_grid = f_train[(n_train+4):,:]
fitp = m.predict(x_grid, F = f_grid)

fitp['mmean']

fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot(x_grid, fitp['mmean'], zorder = 2)
ax.plot(x_grid, fitp['mmean'], zorder = 2)
ax.plot(x_grid, fitp['m_lower'], zorder = 2)
ax.plot(x_grid, fitp['m_upper'], zorder = 2)
plt.show()














#-------------------------

fit1.kernel_
fit1.kernel_.__call__(x1_test)
fit1.kernel_.__call__(x1_train)
fit1.cov(x1_train)
fit1.cov(x1_test)
fit1.cov(x1_test, x1_train)
dir(fit1)

np.sqrt(fit1.cbar_sq_mean_)
np.sqrt(fit1.df_ * fit1.scale_**2 / (fit1.df_ + 2)) # Posterior mode estimate of cbar

k1_out = kernel1.get_params()
k1_out['k1']
kernel1.__call__(x1_test)



# Not what we want
# Get truncation errors for both models -- stop using this chunk
gp_trunc1 = gm.TruncationGP(kernel = kernel1, center=center0, disp=disp0, df=df0, scale=scale0)
gp_trunc2 = gm.TruncationGP(kernel = kernel2, center=center0, disp=disp0, df=df0, scale=scale0)
fit_trunc1 = gp_trunc1.fit(x1_train, y=c1_array, orders=np.arange(0,ns+1))
fit_trunc2 = gp_trunc2.fit(x2_train, y=c2_array, orders=np.arange(0,nl+1))

fit1.predict(x1_train, order=2, return_std=True, kind='trunc')
fit2.predict(x2_train, order=4, return_std=True, kind='trunc') 




#-----------------------------------------------------------
# Paper example
x = np.linspace(0, 1, 100)
X = x[:, None]  # make a 2D version of x to match the input data structure from SciKitLearn
n_orders = 4    # Here we examine the case where we have info on four non-trivial orders
orders = np.arange(0, n_orders)

final_order = 20  # We are going to treat the order-20 result as the final, converged answer
orders_all = np.arange(0, final_order+1)

# The true values of the hyperparameters for generating the EFT coefficients
ls = 0.2
sd = 1
center = 0
ref = 10
ratio = 0.5
nugget = 1e-10
seed = 3

# Get train data
def regular_train_test_split(x, dx_train, dx_test, offset_train=0, offset_test=0, xmin=None, xmax=None):
    train_mask = np.array([(i - offset_train) % dx_train == 0 for i in range(len(x))])
    test_mask = np.array([(i - offset_test) % dx_test == 0 for i in range(len(x))])
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    train_mask = train_mask & (x >= xmin) & (x <= xmax)
    test_mask = test_mask  & (x >= xmin) & (x <= xmax) & (~ train_mask)
    return train_mask, test_mask

x_train_mask, x_valid_mask = regular_train_test_split(
    x, dx_train=24, dx_test=6, offset_train=1, offset_test=1)

kernel = RBF(length_scale=ls, length_scale_bounds='fixed') + \
    WhiteKernel(noise_level=nugget, noise_level_bounds='fixed')
gp = gm.ConjugateGaussianProcess(kernel=kernel, center=center, df=np.inf, scale=sd, nugget=0)

# Draw coefficients and then use `partials` to create the sequence of partial sums
# that defines the order-by-order EFT predictions
# Negative sign to make y_n positive, for aesthetics only
coeffs_all = - gp.sample_y(X, n_samples=final_order+1, random_state=seed)
data_all = gm.partials(coeffs_all, ratio, ref=ref, orders=orders_all)
diffs_all = np.array([data_all[:, 0], *np.diff(data_all, axis=1).T]).T

# Get the "all-orders" curve
data_true = data_all[:, -1]

# Will only consider "known" lower orders for most of the notebook
coeffs = coeffs_all[:, :n_orders]
data = data_all[:, :n_orders]
diffs = diffs_all[:, :n_orders]

# Truncation errors
kernel_fit = RBF(length_scale=ls) + WhiteKernel(noise_level=nugget, noise_level_bounds='fixed')
gp_trunc = gm.TruncationGP(kernel=kernel_fit, ref=ref, ratio=ratio, center=center0, disp=disp0, df=df0, scale=scale0)
fit0 = gp_trunc.fit(X, y=data, orders=orders)
fit0.predict(X, order = 3)



#-------------------------------------------------------------
import numpy as np
from scipy import special, integrate 
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler 

__all__ = ['Models', 'Uncertainties']


class Models():

    def __init__(self, loworder, highorder):

        '''
        The class containing the expansion models from Honda's paper
        and the means to plot them. 
        :Example:
            Models(loworder=np.array([2]), highorder=np.array([5]))
        Parameters:
        -----------
        loworder : numpy.ndarray, int, float
            The value of N_s to be used to truncate the small-g expansion.
            Can be an array of multiple values or one. 
        highorder : numpy.ndarray, int, float
            The value of N_l to be used to truncate the large-g expansion.
            Can be an array of multiple values or one. 
        Returns:
        --------
        None.
        '''

        #check type and assign to class variable
        if isinstance(loworder, float) == True or isinstance(loworder, int) == True:
            loworder = np.array([loworder])

        #check type and assign to class variable
        if isinstance(highorder, float) == True or isinstance(highorder, int) == True:
            highorder = np.array([highorder])

        self.loworder = loworder
        self.highorder = highorder 

        return None 


    def low_g(self, g):
        
        '''
        A function to calculate the small-g divergent asymptotic expansion for a given range in the coupling 
        constant, g.
        
        :Example:
            Models.low_g(g=np.linspace(0.0, 0.5, 20))
            
        Parameters:
        -----------
        g : linspace
            The linspace of the coupling constant for this calculation. 
           
        Returns:
        --------
        output : numpy.ndarray
            The array of values of the expansion in small-g at each point in g_true space, for each value of 
            loworder (highest power the expansion reaches).
        '''

        output = []
        
        for order in self.loworder:
            low_c = np.empty([int(order)+1])
            low_terms = np.empty([int(order) + 1])

            #if g is an array, execute here
            try:
                value = np.empty([len(g)])
       
                #loop over array in g
                for i in range(len(g)):      

                    #loop over orders
                    for k in range(int(order)+1):

                        if k % 2 == 0:
                            low_c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2))
                        else:
                            low_c[k] = 0

                        low_terms[k] = low_c[k] * g[i]**(k) 

                    value[i] = np.sum(low_terms)

                output.append(value)
                data = np.array(output, dtype = np.float64)
            
            #if g is a single value, execute here
            except:
                value = 0.0
                for k in range(int(order)+1):

                    if k % 2 == 0:
                        low_c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2))
                    else:
                        low_c[k] = 0

                    low_terms[k] = low_c[k] * g**(k) 

                value = np.sum(low_terms)
                data = value

        return data

        
    def high_g(self, g):
        
        '''
        A function to calculate the large-g convergent Taylor expansion for a given range in the coupling 
        constant, g.
        
        :Example:
            Models.high_g(g=np.linspace(1e-6,1.0,100))
            
        Parameters:
        -----------
        g : linspace
            The linspace of the coupling constant for this calculation.
            
        Returns
        -------
        output : numpy.ndarray        
            The array of values of the expansion at large-g at each point in g_true space, for each value of highorder
            (highest power the expansion reaches).
        '''

        output = []
        
        for order in self.highorder:
            high_c = np.empty([int(order) + 1])
            high_terms = np.empty([int(order) + 1])
            
            #if g is an array, execute here
            try:
                value = np.empty([len(g)])
        
                #loop over array in g
                for i in range(len(g)):

                    #loop over orders
                    for k in range(int(order)+1):

                        high_c[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                        high_terms[k] = (high_c[k] * g[i]**(-k)) / np.sqrt(g[i])

                    #sum the terms for each value of g
                    value[i] = np.sum(high_terms)

                output.append(value)

                data = np.array(output, dtype = np.float64)
        
            #if g is a single value, execute here           
            except:
                value = 0.0

                #loop over orders
                for k in range(int(order)+1):

                    high_c[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                    high_terms[k] = (high_c[k] * g**(-k)) / np.sqrt(g) 

                #sum the terms for each value of g
                value = np.sum(high_terms)
                data = value
                
        return data 


    def true_model(self, g):
        
        '''
        The true model of the zero-dimensional phi^4 theory partition function using an input linspace.
        
        :Example:
            Models.true_model(g=np.linspace(0.0, 0.5, 100))
            
        Parameters:
        -----------
        g : linspace
            The linspace for g desired to calculate the true model. This can be the g_true linspace, g_data
            linspace, or another linspace of the user's choosing. 
            
        Returns:
        -------
        model : numpy.ndarray        
            The model calculated at each point in g space. 
        '''
    
        #define a function for the integrand
        def function(x,g):
            return np.exp(-(x**2.0)/2.0 - (g**2.0 * x**4.0)) 
    
        #initialization
        self.model = np.zeros([len(g)])
    
        #perform the integral for each g
        for i in range(len(g)):
            
            self.model[i], self.err = integrate.quad(function, -np.inf, np.inf, args=(g[i],))
        
        return self.model 
   

    def plot_models(self, g):
        
        '''
        A plotting function to produce a figure of the model expansions calculated in Models.low_g and Models.high_g, 
        and including the true model calculated using Mixing.true_model.
        
        :Example:
            Mixing.plot_models(g=np.linspace(0.0, 0.5, 100))
            
        Parameters:
        -----------
        g : linspace
            The linspace in on which the models will be plotted here. 
        Returns
        -------
        None.
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)

        #x and y limits
        if min(g) == 1e-6:
            ax.set_xlim(0.0, max(g))
        else:
            ax.set_xlim(min(g), max(g))
        ax.set_ylim(1.2,3.2)
        ax.set_yticks([1.2, 1.6, 2.0, 2.4, 2.8, 3.2])
  
        #plot the true model 
        ax.plot(g, self.true_model(g), 'k', label='True model')
        
        #add linestyle cycler
        linestyle_cycler = cycler(linestyle=['dashed', 'dotted', 'dashdot', 'dashed', 'dotted', 'dashdot'])
        ax.set_prop_cycle(linestyle_cycler)
                
        #for each large-g order, calculate and plot
        for i,j in zip(range(len(self.loworder)), self.loworder):
            ax.plot(g, self.low_g(g)[i,:], color='r', label=r'$f_s$ ($N_s$ = {})'.format(j))

        #for each large-g order, calculate and plot
        for i,j in zip(range(len(self.highorder)), self.highorder):
            ax.plot(g, self.high_g(g)[i,:], color='b', label=r'$f_l$ ($N_l$ = {})'.format(j))
                    
        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        #save figure option
        # response = input('Would you like to save this figure? (yes/no)')

        # if response == 'yes':
        #     name = input('Enter a file name (include .jpg, .png, etc.)')
        #     fig.savefig(name, bbox_inches='tight')

        return None
        
         
    def residuals(self):
        
        '''
        A calculation and plot of the residuals of the model expansions vs the true model values at each point in g.
        g is set internally for this plot, as the plot must be shown in loglog format to see the power law of the
        residuals. 
        
        :Example:
            Mixing.residuals()
            
        Parameters:
        -----------
        None.
                  
        Returns:
        --------
        None. 
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('Residual', fontsize=22)
        ax.set_title('F(g): residuals', fontsize=22)
        ax.set_xlim(1e-2, 10.)
        ax.set_ylim(1e-6, 1e17)

        #set range for g
        g_ext = np.logspace(-6., 6., 800)
        
        #set up marker cycler
        marker_cycler = cycler(marker=['.', '*', '+', '.', '*', '+'])
        ax.set_prop_cycle(marker_cycler)

        #calculate true model
        value_true = self.true_model(g_ext)
        
        #for each small-g order, plot
        valuelow = np.zeros([len(self.loworder), len(g_ext)])
        residlow = np.zeros([len(self.loworder), len(g_ext)])

        for i,j in zip(range(len(self.loworder)), self.loworder):
            valuelow[i,:] = self.low_g(g_ext)[i]
            residlow[i,:] = (valuelow[i,:] - value_true)/value_true
            ax.loglog(g_ext, abs(residlow[i,:]), 'r', linestyle="None", label=r"$F_s({})$".format(j))

        #for each large-g order, plot
        valuehi = np.zeros([len(self.highorder), len(g_ext)])
        residhi = np.zeros([len(self.highorder), len(g_ext)])

        for i,j in zip(range(len(self.highorder)), self.highorder):
            valuehi[i,:] = self.high_g(g_ext)[i]
            residhi[i,:] = (valuehi[i,:] - value_true)/value_true
            ax.loglog(g_ext, abs(residhi[i,:]), 'b', linestyle="None", label=r"$F_l({})$".format(j))
        
        ax.legend(fontsize=18)
        plt.show()


class Uncertainties:


    def __init__(self, error_model='informative'):

        '''
        An accompanying class to Models() that possesses the truncation error models
        that are included as variances with the small-g and large-g expansions. 
        :Example:
            Uncertainties()
        Parameters:
        -----------
        error_model : str
            The name of the error model to use in the calculation. Options are
            'uninformative' and 'informative'. Default is 'informative'.
        Returns:
        --------
        None.
        '''

        #assign error model 
        if error_model == 'uninformative':
           self.error_model = 1

        elif error_model == 'informative':
            self.error_model = 2

        else:
            raise ValueError("Please choose 'uninformative' or 'informative'.")

        return None


    def variance_low(self, g, loworder):


        '''
        A function to calculate the variance corresponding to the small-g expansion model.
        :Example:
            Bivariate.variance_low(g=np.linspace(1e-6, 0.5, 100), loworder=5)
        Parameters:
        -----------
        g : numpy.linspace
            The linspace over which this calculation is performed.
        loworder : int
            The order to which we know our expansion model. Must be passed one at a time if
            more than one model is to be calculated.
        Returns:
        --------
        var1 : numpy.ndarray
            The array of variance values corresponding to each value in the linspace of g. 
        '''

        #even order 
        if loworder % 2 == 0:
            
            #find coefficients
            c = np.empty([int(loworder + 2)])

            #model 1 for even orders
            if self.error_model == 1:

                for k in range(int(loworder + 2)):

                    if k % 2 == 0:
                        c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k) * math.factorial(k//2))
                    else:
                        c[k] = 0.0

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance 
                var1 = (cbar)**2.0 * (math.factorial(loworder + 2))**2.0 * g**(2.0*(loworder + 2))

            #model 2 for even orders
            elif self.error_model == 2:

                for k in range(int(loworder + 2)):

                    if k % 2 == 0:

                        #skip first coefficient
                        if k == 0:
                            c[k] = 0.0
                        else:
                            c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2) \
                                   * math.factorial(k//2 - 1) * 4.0**(k))
                    else:
                        c[k] = 0.0

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial(loworder//2))**2.0 * (4.0 * g)**(2.0*(loworder + 2))

        #odd order
        else:

            #find coefficients
            c = np.empty([int(loworder + 1)])

            #model 1 for odd orders
            if self.error_model == 1:

                for k in range(int(loworder + 1)):

                    if k % 2 == 0:
                        c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k) * math.factorial(k//2))
                    else:
                        c[k] = 0.0

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial(loworder + 1))**2.0 * g**(2.0*(loworder + 1))

            #model 2 for odd orders
            elif self.error_model == 2:

                for k in range(int(loworder + 1)):

                    if k % 2 == 0:

                        #skip first coefficient
                        if k == 0:
                            c[k] = 0.0
                        else:
                            c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2) \
                                    * math.factorial(k//2 - 1) * 4.0**(k))
                    else:
                        c[k] = 0.0

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial((loworder-1)//2))**2.0 * (4.0 * g)**(2.0*(loworder + 1))

        return var1


    def variance_high(self, g, highorder):

        '''
        A function to calculate the variance corresponding to the large-g expansion model.
        :Example:
            Bivariate.variance_low(g=np.linspace(1e-6, 0.5, 100), highorder=23)
        Parameters:
        -----------
        g : numpy.linspace
            The linspace over which this calculation is performed.
        highorder : int
            The order to which we know our expansion model. This must be a single value.
            
        Returns:
        --------
        var2 : numpy.ndarray
            The array of variance values corresponding to each value in the linspace of g. 
        '''

        #find coefficients
        d = np.zeros([int(highorder) + 1])

        #model 1
        if self.error_model == 1:

            for k in range(int(highorder) + 1):

                d[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k * (math.factorial(k)) / (2.0 * math.factorial(k))

            #rms value (ignore first two coefficients in this model)
            dbar = np.sqrt(np.sum((np.asarray(d)[2:])**2.0) / (highorder-1))

            #variance
            var2 = (dbar)**2.0 * (g)**(-1.0) * (math.factorial(highorder + 1))**(-2.0) \
                    * g**(-2.0*highorder - 2)

        #model 2
        elif self.error_model == 2:

            for k in range(int(highorder) + 1):

                d[k] = special.gamma(k/2.0 + 0.25) * special.gamma(k/2.0 + 1.0) * 4.0**(k) \
                       * (-0.5)**k / (2.0 * math.factorial(k))

            #rms value
            dbar = np.sqrt(np.sum((np.asarray(d))**2.0) / (highorder + 1))

            #variance
            var2 = (dbar)**2.0 * g**(-1.0) * (special.gamma((highorder + 3)/2.0))**(-2.0) \
                    * (4.0 * g)**(-2.0*highorder - 2.0)

        return var2



eft_unc = Uncertainties(error_model='uninformative')

# Using grid of 20 points
eft_unc.variance_low(g = np.linspace(0.03, 0.5, 20), loworder=4)
eft_unc.variance_high(g = np.linspace(0.03, 0.5, 20), highorder=4)

# Using grid of 4 points
x_eft_train = np.array([0.05,0.2,0.3,0.45])
eft_unc.variance_low(g = x_eft_train, loworder=2)
eft_unc.variance_low(g = x_eft_train, loworder=4)
eft_unc.variance_low(g = x_eft_train, loworder=6)

eft_unc.variance_high(g = x_eft_train, highorder=4)

eft_models = Models(3,3)
eft_models.low_g(np.linspace(0.03, 0.5, 20))
eft_models.high_g(np.linspace(0.03, 0.5, 20))



x_train = np.linspace(0.03, 0.5, 20)
p2 = 1/eft_unc.variance_low(g = x_train, loworder=2)
p4 = 1/eft_unc.variance_low(g = x_train, loworder=4)
p6 = 1/eft_unc.variance_low(g = x_train, loworder=6)

p2/(p2+p4+p6)


eft_unc.variance_low(g = x_train, loworder=3)
eft_unc.variance_high(g = x_train, highorder=3)