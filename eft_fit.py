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
x1_train = np.linspace(0.05, 0.5, 5)
x2_train = np.linspace(0.05, 0.5, 5)

x1_test = np.linspace(0.03, 0.48, 10)
x2_test = np.linspace(0.03, 0.48, 10)

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
df0 = 10000
scale0 = 0.0001
kernel1 = RBF(length_scale=0.3, length_scale_bounds=(0.05, 4)) + WhiteKernel(1e-10, noise_level_bounds='fixed')
kernel2 = RBF(length_scale=0.5, length_scale_bounds=(0.05, 1)) + WhiteKernel(1e-10, noise_level_bounds='fixed')

exp1_res = get_exp(list(chain(*x1_data.tolist())), 2, 'sg')
y1_df = exp1_res['ypred_df']
coeffs1 = gm.coefficients(np.array(y1_df),ratio = 1,orders=np.arange(0,ns+1))

gp1 = gm.ConjugateGaussianProcess(kernel1, center=center0, disp=disp0, df=df0, scale=5, n_restarts_optimizer=10)
fit1 = gp1.fit(x1_train, coeffs1[:5,])
fit1.predict(x1_test,return_std=True)
np.sum(fit1.predict(x1_test,return_std=False),axis = 1)


ratio1 = np.array(list(chain(*exp1_res['qx_df'].values.tolist()))) # Reshape the array
ref1 = np.array(list(chain(*exp1_res['yref_df'].values.tolist()))) # Reshape the array

gp_trunc1 = gm.TruncationGP(kernel = fit1.kernel_, center=center0, disp=disp0, df=df0, scale=scale0, ref = ref1, ratio = ratio1)
fit_trunc1 = gp_trunc1.fit(x1_data, y=np.array(exp1_res['ypred_df']), orders=np.arange(0,ns+1))
fit_trunc1.predict(x1_data, order = 2, return_std = True)

fhat = fit_trunc1.predict(x1_data, order = 2, return_std = False)
std = fit_trunc1.predict(x1_data, order = 2, return_std = True)[1]

fig, ax = plt.subplots(figsize=(3.2, 3.2))
fig.plot(x1_test, fhat[5:], zorder = 2)
ax.fill_between(x1_test[:,0], fhat[5:] + 2*std[5:], fhat[5:] - 2*std[5:], zorder=1)
plt.show()


exp2_res = get_exp(list(chain(*x2_data.tolist())), 4, 'lg')
y2_df = exp2_res['ypred_df']

ratio2 = np.array(list(chain(*exp2_res['qx_df'].values.tolist()))) # Reshape the array
ref2 = np.array(list(chain(*exp2_res['yref_df'].values.tolist()))) # Reshape the array

coeffs2 = gm.coefficients(np.array(y2_df),ratio = ref2,orders=np.arange(0,nl+1))

gp2 = gm.ConjugateGaussianProcess(kernel2, center=center0, disp=disp0, df=df0, scale=5, n_restarts_optimizer=10)
fit2 = gp2.fit(x2_train, coeffs2[:5,])
fit2.predict(x2_test,return_std=True)
np.sum(fit2.predict(x2_test,return_std=False),axis = 1)

gp_trunc2 = gm.TruncationGP(kernel = fit2.kernel_, center=center0, disp=disp0, df=df0, scale=scale0, ref = ref2, ratio = ratio2)
fit_trunc2 = gp_trunc2.fit(x2_data, y=np.array(exp2_res['ypred_df']), orders=np.arange(0,nl+1))
fit_trunc2.predict(x2_data, order = 4, return_std = True)

fhat = fit_trunc2.predict(x2_data, order = 4, return_std = False)
std = fit_trunc2.predict(x2_data, order = 4, return_std = True)[1]

fig, ax = plt.subplots(figsize=(3.2, 3.2))
fig.plot(x2_test, fhat[5:], zorder = 2)
ax.fill_between(x1_test[:,0], fhat[5:] + 2*std[5:], fhat[5:] - 2*std[5:], zorder=1)
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