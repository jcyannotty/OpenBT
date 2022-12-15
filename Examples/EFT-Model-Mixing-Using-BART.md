EFT Model Mixing using BART
================
John Yannotty
11-11-22

<style type="text/css">
  body{
  line-height: 1.5
  }
</style>
# Introduction

This document provides a brief tutorial on how to perform model mixing
with Bayesian Additive Regression Trees (BART) and serves as an online
supplement to the article “Model Mixing Using Bayesian Additive
Regression Trees”. The three examples shown below are described in more
detail by Yannotty et al. (2022).

This code contributes to the Open Bayesian Tree repository. The original
code can be found
[mpratola/openbt](https://bitbucket.org/mpratola/openbt/src/master/)
while an updated repository which includes the model mixing code is
found at [jcyannotty/openbt](https://github.com/jcyannotty/OpenBT).

## Installation and Setup

This section outlines the steps for downloading the files required for
model mixing with BART.

1.  Download the most recent OpenBT Ubuntu Linux 20.04 package from
    GitHub:
    [jcyannotty/openbt/package](https://github.com/jcyannotty/OpenBT/blob/main/openbt_0.current_amd64-MPI_Ubuntu_20.04.deb).
    This can alternatively be done by using the command line as shown
    below. The first argument after `-O` specifies the folder where the
    packge is downloaded while the second argument specifies the
    location of the package. on Github

        $ wget -O /location/of/downloaded/openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb
        https://github.com/jcyannotty/OpenBT/raw/main/openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb

2.  Install the package locally using the command line. Please make sure
    to substitute the location where the .deb package is stored into
    “/location/of/downloaded/.deb”:

        $ cd /location/of/downloaded/.deb

        $ sudo apt-get install ./openbt_0.current_amd64-MPI_Ubuntu_20.04.deb

3.  Source in the required R scripts from GitHub, which are located at
    [jcyannotty/openbt](https://github.com/jcyannotty/OpenBT). The
    following lines of R code can be used to read in these files.

        # Read in openbt R functions
        source("https://github.com/jcyannotty/OpenBT/blob/main/src/openbt.R?raw=TRUE")

        # Read in Honda EFT functions:  
        source("https://github.com/jcyannotty/OpenBT/blob/main/R/honda_eft.R?raw=TRUE")

        # Model Mixing Helper functions
        source("https://github.com/jcyannotty/OpenBT/blob/main/R/eft_mixing_helper_functions.R?raw=TRUE")

4.  The code for running the examples shown in this document can be
    found at
    [jcyannotty/openbt/eft_examples](https://github.com/jcyannotty/OpenBT/blob/main/R/eft_examples.R).
    This can also be done with the command line as follows.

        $ wget -O /location/of/downloaded/eft_examples.R https://github.com/jcyannotty/OpenBT/raw/main/R/eft_examples.R

## OpenBT Code

The Bayesian tree code is written in C++. Interfaces are written in R
and Python which easily allows the user to execute the C++ code. This
tutorial focuses on the R implementation of the software.

Only three functions are required for model mixing: `openbt(...)`,
`predict.openbt(...)`, and `mixingwts.openbt(...)`. The essential
arguments for each function are listed below.

#### `openbt(...)`

- x.train: (n x p matrix; n = sample size, p = number of predictors) The
  design matrix of training inputs.
- y.train: (n-dimensional vector): The vector of observational data.
- f.train: (n x K matrix; n = sample size, K = number of models) The
  mean predictions from each individual model. Each columns corresponds
  to an individual model’s predictions.
- f.sd.train: (n x K matrix) The estimated standard deviations for each
  model which are used in the informative prior. If this argument is not
  passed, then the non-informative prior will be used.
- model: (string) The type of Bayesian tree model to be fit. Specify
  “mixbart” for model mixing with BART.
- pbd: (double) The probability of the birth or death moves in the
  weight tree model. A two-dimensional vector is passed to control the
  birth and death for a heteroscedastic tree model.
- ntree: (int) The number of trees for the weight functions.
- k: (double, \>0) The tuning parameter in the prior variance of the
  weights.
- overallsd: (double, \>0): An initial estimate of the error standard
  deviation. This is used to calibrate
  ![\lambda](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Clambda "\lambda").
- overallnu: (double,
  ![\small{\geq 3}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Cgeq%203%7D "\small{\geq 3}"))
  The shape parameter for the error variance prior.
- power: (double, \>0) The power parameter in the tree prior.
- base: (double, between 0 and 1) The base parameter in the tree prior.
- numcut: (int) The number of cutpoints to use per predictor. The
  cutpoints are used in the splitting rules for the trees.
- tc: (int) The number of cores to use when fitting the model, essential
  for parallelization with the MPI.
- minnumbot (int): The minimum number of observations that are assigned
  to a terminal node.
- ndpost: (int) The number of posterior samples to keep.
- nadapt: (int) The number of adapt steps to use in the MCMC. These
  steps are not included in the final posterior and are only used to
  tune the proposal distributions in the Metropolis-Hastings algorithm.
- adaptevery: (int) The number of samples between adapt steps.
- nskip: (int) Number of burn-in steps for the MCMC.
- printevery: (int) Controls how often the progress should be printed
  when fitting the model.
- modelname: (str) Results are stored using this name after the training
  process.

Output: a list of the model features (such as tuning parameter values
and file locations). The MCMC draws from the tree models is written to a
text file and stored in a temporary directory. The directory is
specified in the “folder” object in the output list from the
`openbt(...)` function.

#### `predict.openbt(...)`

- fit: (openbt object) The output from the openbt(…) function.
- x.test: (ntest x K matrix; ntest = number of test points, K = number
  of models) The design matrix for the test inputs.
- f.test: (ntest x K matrix) The matrix of mean predictions from each
  individual model at the grid of test inputs.
- tc: (int) The number of cores to use for parallelization when getting
  predictions.
- q.lower: (double, between 0 and 1) The lower quantile for the credible
  interval.
- q.upper: (double, between 0 and 1) The upper quantile for the credible
  interval.

Output: a list of the posterior draws of the predicted mixed-mean
function and error standard deviations
(![\small{\sigma_i}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Csigma_i%7D "\small{\sigma_i}"))
at each of the test points along with summary statistics. Note, the
posteriors of the standard deviations will be the same across all test
points when using a homoscedastic variance.

#### `mixingwts.openbt(...)`

- fit: (openbt object) The output from the openbt(…) function.
- x_test: (ntest x K matrix; ntest = number of test points, K = number
  of models) The design matrix for the test inputs.
- numwts (int): The number of weights functions in the model. Should be
  equal to K.  
- tc: (int) The number of cores to use for parallelization when getting
  predictions.
- q.lower: (double, between 0 and 1) The lower quantile for the credible
  interval.
- q.upper: (double, between 0 and 1) The upper quantile for the credible
  interval.

Output: a list of the posterior draws of the predicted weight functions
at each of the test points along with summary statistics.

# Effective Field Theories

This work is motivated by problems in nuclear physics which are modeled
using a technique known as Effective Field Theory (EFT). An EFT models
physical systems with an infinite expansion of terms organized in order
of decreasing importance according to the power counting principle
(Melendez, 2019). In practice, only a finite number of lower-order terms
are known. Thus, the theoretical predictions from an EFT at a
d-dimensional input
![x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x "x")
are decomposed into two terms using an additive model  

![f\_\dagger(x) = h^{(N)}(x) + \delta^{(N)}(x),](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_%5Cdagger%28x%29%20%3D%20h%5E%7B%28N%29%7D%28x%29%20%2B%20%5Cdelta%5E%7B%28N%29%7D%28x%29%2C "f_\dagger(x) = h^{(N)}(x) + \delta^{(N)}(x),")

where
![f\_\dagger(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_%5Cdagger%28x%29 "f_\dagger(x)")
denotes the theoretical prediction of the system,
![h^{(N)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;h%5E%7B%28N%29%7D%28x%29 "h^{(N)}(x)")
denotes a finite-order expansion of order N, and
![\delta^{(N)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cdelta%5E%7B%28N%29%7D%28x%29 "\delta^{(N)}(x)")
denotes the truncation error. The truncation error is a Gaussian Process
(GP) or a pointwise error model (Melendez, 2019). A more detailed
discussion about EFTs is found in the additional supplementary material.

Prototypes of such models are the weak and strong coupling finite-order
expansions for the partition function of the zero-dimensional
![\phi^4](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cphi%5E4 "\phi^4")
theory presented by Honda (2014). In this example, the true physical
system is defined by where
![x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x "x")
denotes the coupling constant. Two types of finite-order expansions
exist for this partition function and are given below for
![n_s](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;n_s "n_s")
or
![n_l](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;n_l "n_l")
![\geq 1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cgeq%201 "\geq 1").

One may consider a set of EFTs to recover the underlying system. Three
examples of model sets are shown in Figure 1. These models represent the
general features of EFTs in that they offer high fidelity predictions in
one region of the domain and diverge elsewhere.

<img src="EFT-Model-Mixing-Using-BART_files/figure-gfm/Plot Expansion-1.png" style="display: block; margin: auto;" />

The localized predictive accuracy of each EFT is also described by its
estimated truncation error. As the finite-order expansion diverges, the
variance of the truncation grows. Hence, this variance is a reliable
indicator as to where the finite-order expansion can be trusted to
accurately describe the underlying system. The GP model for the
truncation error depends on physical quantities
![\small{Q(x)}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7BQ%28x%29%7D "\small{Q(x)}")
and
![\small{y\_{\text{ref}}(x)}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7By_%7B%5Ctext%7Bref%7D%7D%28x%29%7D "\small{y_{\text{ref}}(x)}"),
which are described in the additional supplementary material. The
specification of these quantities is generally dependent on domain
expertise and directly affects the estimation of the truncation error
variance. Since these quantities are unknown for our motivating example,
we elect to use the pointwise error approximation as outlined by
Semposki et al. (2022).

# Examples

This section briefly discusses the results obtained when mixing the mean
predictions from the EFTs across the three experimental settings
mentioned above. Additionally, we are also interested in highlighting
the differences in the results obtained between the two priors. Both
priors lead to adequate mean predictions of the true underlying physical
system, however the informative prior generally results in less
uncertainty. The code to mix the mean predictions from each set of EFTs
using BART is provided below. In each case, we present the code for the
non-informative prior first followed by the code to perform model mixing
with an informative prior.

Note, the results presented below are similar to those presented in the
paper. One can expect subtle differences in the results each time the
MCMC is run. This is more apparent especially when using a lower amount
of trees in the model because each tree is more influential on the
results.

## Example 1

This first example considers mixing the 2nd order weak coupling
expansion and 4th order strong coupling expansion, which are shown in
panel (a) of Figure 1. The true system lies between the two EFTs, hence
we can expect the mixed model to roughly interpolate between their
corresponding mean predictions.

### Training the Model

This section illustrates the code to perform model mixing with BART. The
first subsection of this code chunk trains the model, gets predictions
of the system, and generates the posterior of the weight functions when
using the non-informative prior. The second subsection yields the same
output under the informative prior. To use the informative prior, pass
in the matrix of prediction standard deviations to the argument
`f.sd.train`. If this argument is not specified, then the
non-informative prior is used.

``` r
#-----------------------------------------------------
# Non-Informative Prior
#-----------------------------------------------------
# Attach the data set
attach(ex1_data)

# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit1=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),model="mixbart",
           ntree = 8,k = 2.5, overallsd = sqrt(sig2_hat), overallnu = 5,power = 2.0, base = 0.95,
           ntreeh=1,numcut=300,tc=4,minnumbot = 4,
           ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 1000,
           summarystats = FALSE,modelname="eft_mixing")

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp1=predict.openbt(fit1,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw1=mixingwts.openbt(fit1, x.test = x_test, numwts = 2, tc = 4, q.lower = 0.025, q.upper = 0.975)


#-----------------------------------------------------
# Informative Prior
#-----------------------------------------------------
# Perform model mixing with the Non-informative prior.
# Note, f.sd.train IS specified, hence the informative prior is used.
fit2=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),f.sd.train = f_train_dsd,model="mixbart",
           ntree = 10,k = 5, overallsd = sqrt(sig2_hat), overallnu = 8, power = 2.0, base = 0.95, 
           ntreeh=1,numcut=300,tc=4,minnumbot = 3,
           ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 500,
           modelname="eft_mixing"
          )

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp2=predict.openbt(fit2,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw2=mixingwts.openbt(fit2, x.test = x_test, numwts = 2, tc = 4, q.lower = 0.025, q.upper = 0.975)
```

### Posterior Mean Predictions

The mixed mean predictions (purple line) and associated 95% credible
intervals (purple shade) are shown in Figure 2. Regardless of the prior
selected, the mean predictions adequately recover the true system when
using either prior. The main difference in the results is seen in the
estimated uncertainty within the intermediate range as the informative
prior offers a more narrow credible intervals compared to those obtained
with the non-informative prior. This is to be expected because more
information about the weight functions is directly embedded into the
resulting posterior distributions when using the informative prior.

#### Results

<img src="EFT-Model-Mixing-Using-BART_files/figure-gfm/Ex1 Predictions-1.png" style="display: block; margin: auto;" />

#### Code

``` r
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the predictions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set the labels for hte legend
g_labs = c(expression(paste(hat(f)[1], '(x)')), expression(paste(hat(f)[2], '(x)')),
           expression(paste(f["\u2020"],'(x)')),"Post. Mean")
# Plot the predictions obtained with the non-informative prior
p1 = plot_fit_gg2(ex1_data, fitp1, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.85,2.75), title = "Non-Informative Prior", grid_on = TRUE)
# Plot the predictions obtained with the informative prior
p2 = plot_fit_gg2(ex1_data, fitp2, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.85,2.75), title = "Informative Prior", grid_on = TRUE)
# Resize text elements
p1 = p1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
p2 = p2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure
legend1 = g_legend(p1)
grid.arrange(arrangeGrob(p1 + theme(legend.position = "none"),
                         p2 + theme(legend.position = "none"),
                         nrow = 1), nrow=2, heights = c(10,1), legend = legend1,
                         top = textGrob("Figure 2: Posterior Mean Predictions", gp = gpar(fontsize = 16)))
```

### Posterior Weight Functions

The corresponding weight functions are shown below in Figure 3. As
expected, the weight functions under the informative prior exhibit
significantly lower amounts of uncertainty. Nonetheless, both approaches
yield similar mean predictions of the weight functions. Common features
of the weight functions include:

- The functions intersect between
  ![x = 0.25](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x%20%3D%200.25 "x = 0.25")
  and
  ![x = 0.28](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x%20%3D%200.28 "x = 0.28")
  at a weight value close to 0.5. This indicates that the inflection
  point between the predictive strengths of the two EFTs occurs
  somewhere between
  ![(0.25,0.28)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%280.25%2C0.28%29 "(0.25,0.28)").
- The weights generally stay within a range of
  ![\[0,1\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5B0%2C1%5D "[0,1]").
  This results from (1) the prior regularization and (2) the true system
  lies between the functions - hence an interpolation of the mean
  predictions is a useful (and perhaps natural) solution.

#### Results

<img src="EFT-Model-Mixing-Using-BART_files/figure-gfm/Ex1 Weights-1.png" style="display: block; margin: auto;" />

#### Code

``` r
#------------------------------------------------
# Plot the weight functions
#------------------------------------------------
# Create the plot for weights under the non-informative prior
w1 = plot_wts_gg2(fitw1, ex1_data$x_test, y_lim = c(-0.05, 1.05),title = 'Non-Informative Prior', colors = color_list,
                  line_type_list = lty_list, gray_scale = FALSE)
# Create the plot for weights under the informative prior
w2 = plot_wts_gg2(fitw2, ex1_data$x_test, y_lim = c(-0.05, 1.05), title = 'Informative Prior',colors = color_list,
                  line_type_list = lty_list, gray_scale = FALSE)
# Resize text elements
w1 = w1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
w2 = w2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots 
legend_w1 = g_legend(w1)
grid.arrange(arrangeGrob(w1 + theme(legend.position = "none"),
                         w2 + theme(legend.position = "none"),
                         nrow = 1),nrow=2, heights = c(10,1), legend = legend_w1,
                        top = textGrob("Figure 3: Posterior Weight Functions", gp = gpar(fontsize = 16)))
```

## Example 2

Now consider mixing the weak and strong coupling expansions shown in
Figure 1(b). The mean predictions from these two EFTs are convex
functions which overestimate the true system in the intermediate range
of the domain. Hence, an interpolation of these models is insufficient
for recovering the true system. Since the BART-based approach does not
place any strict constraints on the weight functions, we are able to
leverage the information in the data and recover the true system within
this part of the domain.

### Training the Model

``` r
#-----------------------------------------------------
# Non-Informative Prior
#-----------------------------------------------------
# Attach example 2 data
attach(ex2_data)

# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit1=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),model="mixbart",
           ntree = 8,k = 4.0, overallsd = sqrt(sig2_hat), overallnu = 5,power = 2.0, base = 0.95,
           ntreeh=1,numcut=300,tc=4,minnumbot = 4,
           ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 1000,
           summarystats = FALSE,modelname="eft_mixing")

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp1=predict.openbt(fit1,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw1=mixingwts.openbt(fit1, x.test = x_test, numwts = 2, tc = 4, q.lower = 0.025, q.upper = 0.975)


#-----------------------------------------------------
# Informative Prior
#-----------------------------------------------------
# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train IS specified, hence the informative prior is used.
fit2=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),f.sd.train = f_train_dsd,model="mixbart",
           ntree = 10,k = 5, overallsd = sqrt(sig2_hat), overallnu = 5, power = 2.0, base = 0.95, 
           ntreeh=1,numcut=300,tc=4,minnumbot = 3,
           ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 500,
           modelname="eft_mixing"
          )

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp2=predict.openbt(fit2,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw2=mixingwts.openbt(fit2, x.test = x_test, numwts = 2, tc = 4, q.lower = 0.025, q.upper = 0.975)

# Attach example 2 data
detach(ex2_data)
```

### Posterior Mean Predictions

The mean predictions from the mixed model are shown below. Similar to
before, the true system is recovered by the posterior mean regardless of
the prior selected. The corresponding uncertainty is lower when using
the informative prior as opposed to the non-informative prior.

#### Results

<img src="EFT-Model-Mixing-Using-BART_files/figure-gfm/Ex2 Predictions-1.png" style="display: block; margin: auto;" />

#### Code

``` r
#------------------------------------------------
# Plot the predictions
#------------------------------------------------
# Set the labels in the legend
g_labs = c(expression(paste(hat(f)[1], '(x)')), expression(paste(hat(f)[2], '(x)')),
           expression(paste(f["\u2020"],'(x)')),
           "Post. Mean")
# Plot the predictions from the non-informative prior
p1 = plot_fit_gg2(ex2_data, fitp1, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.9,2.8), grid_on = TRUE, title = "Non-Informative Prior")

# Plot the predictions from the informative prior
p2 = plot_fit_gg2(ex2_data, fitp2, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.9,2.8), title = "Informative Prior", grid_on = TRUE)

# Resize the plot text
p1 = p1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
p2 = p2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots
legend1 = g_legend(p1)
grid.arrange(arrangeGrob(p1 + theme(legend.position = "none"),
                         p2 + theme(legend.position = "none"),
                         nrow = 1), nrow=2, heights = c(10,1), legend = legend1,
                         top = textGrob("Posterior Mean Predictions", gp = gpar(fontsize = 16)))
```

### Posterior Weight Functions

The corresponding posterior weight functions are shown below. The
weights take a sigmoid-like shape, which is intuitive due to the
contrasting predictive performances of both EFTs. Similar to the first
example, the weight functions exhibit less uncertainty under the
informative prior compared to the informative prior. One interesting
thing to note is that the weights intersect around 0.27, however this
time the weight value is below 0.45. This suggests a strict constraint,
such as a sum-to-one constraint, is inappropriate for this problem.

#### Results

<img src="EFT-Model-Mixing-Using-BART_files/figure-gfm/Ex2 Weights-1.png" style="display: block; margin: auto;" />

#### Code

``` r
#------------------------------------------------
# Plot the weight functions
#------------------------------------------------
# Create the plot for weights under the non-informative prior
w1 = plot_wts_gg2(fitw1, ex2_data$x_test, y_lim = c(-0.05, 1.05),title = 'Non-Informative Prior',
                  colors = color_list, line_type_list = lty_list, gray_scale = FALSE)
# Create the plot for the weights under the informative prior
w2 = plot_wts_gg2(fitw2, ex2_data$x_test, y_lim = c(-0.05, 1.05), title = 'Informative Prior',
                  colors = color_list, line_type_list = lty_list, gray_scale = FALSE)
# Resize text elements
w1 = w1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
w2 = w2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots
legend_w1 = g_legend(w1)
grid.arrange(arrangeGrob(w1 + theme(legend.position = "none"),
                         w2 + theme(legend.position = "none"),
                         nrow = 1),nrow=2, heights = c(10,1), legend = legend_w1,
                        top = textGrob("Posterior Weight Functions", gp = gpar(fontsize = 16)))
```

# Example 3:

Finally, consider the case where the model set does not contain a local
expert in a specific region. In other words, none of the models
accurately predict the true system within this region. This set of EFTs
is displayed in Figure 1(c). In this example, we must leverage
information from the data in the right portion of the domain to
accurately predict the system.

### Training the Model

``` r
#-----------------------------------------------------
# Non-Informative Prior
#-----------------------------------------------------
# Attach example 2 data
attach(ex3_data)

# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit1=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),model="mixbart",
           ntree = 10,k = 5.5, overallsd = sqrt(sig2_hat), overallnu = 5,power = 2.0, base = 0.95,
           ntreeh=1,numcut=300,tc=4,minnumbot = 4,
           ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 1000,
           summarystats = FALSE,modelname="eft_mixing")

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp1=predict.openbt(fit1,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw1=mixingwts.openbt(fit1, x.test = x_test, numwts = 3, tc = 4, q.lower = 0.025, q.upper = 0.975)


#-----------------------------------------------------
# Informative Prior
#-----------------------------------------------------
# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train IS specified, hence the informative prior is used.
fit2=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),f.sd.train = f_train_dsd,model="mixbart",
           ntree = 8, k = 6.5, overallsd = sqrt(sig2_hat), overallnu = 5, power = 2.0, base = 0.95, 
           ntreeh=1,numcut=300,tc=4,minnumbot = 3,
           ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 500,
           modelname="eft_mixing"
          )

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp2=predict.openbt(fit2,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw2=mixingwts.openbt(fit2, x.test = x_test, numwts = 3, tc = 4, q.lower = 0.025, q.upper = 0.975)

# Attach example 2 data
detach(ex3_data)
```

### Posterior Mean Predictions

The mean predictions from the mixed model under both priors are shown
below. Again, both approaches yield accurate mean predictions, while the
informative prior results in more narrow credible intervals. Under both
priors, lower amounts of trees generally results in smoother and narrow
uncertainty intervals across the majority of the domain. However, with a
lower amount of trees, the exact locations of the larger uncertainty
bands might slightly change depending on how the trees split during the
MCMC. Introducing more trees (more than 20) results in more uncertainty
between training points, however the results are less influenced by an
individual tree.

#### Results

<img src="EFT-Model-Mixing-Using-BART_files/figure-gfm/Ex3 Predictions-1.png" style="display: block; margin: auto;" />

#### Code

``` r
#------------------------------------------------
# Plot the predictions
#------------------------------------------------
# Set the labels in the legend
g_labs = c(expression(paste(hat(f)[1], '(x)')), expression(paste(hat(f)[2], '(x)')),
           expression(paste(hat(f)[3], '(x)')), expression(paste(f["\u2020"],'(x)')),
           "Post. Mean")
# Generate the first plot
p1 = plot_fit_gg2(ex3_data, fitp1, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.8,2.7), title = "Non-Informative Prior", grid_on = TRUE)
# Generate the first plot
p2 = plot_fit_gg2(ex3_data, fitp2, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.8,2.7), title = "Informative Prior", grid_on = TRUE)
# Resize plot text
p1 = p1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
p2 = p2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots
legend1 = g_legend(p1)
grid.arrange(arrangeGrob(p1 + theme(legend.position = "none"),
                         p2 + theme(legend.position = "none"),
                         nrow = 1), nrow=2, heights = c(10,1), legend = legend1,
                         top = textGrob("Posterior Mean Predictions", gp = gpar(fontsize = 16)))
```

### Posterior Weight Functions

The posterior weight functions are shown below for this third example.
The main takeaway from this figure is that the BART-based mixing
approach searches for useful combinations of the mean predictions to
recover the true mean. Thus, the solution to this problem is not unique,
meaning multiple sets of weight functions can be used to recover the
true system. With the non-informative prior, the effect of the 6th order
weak coupling expansion (green) is shrunk towards zero. This occurs even
in the left portion of the domain despite its accurate predictions of
the true system. This likely occurs because the trees are regularized to
be weak learners which implies each tree maintains a shallow structure.
Because of this, the contribution from each tree to the weight functions
still maintains a sense of global model performance. Meanwhile, with the
informative prior, the 6th order expansion is given the highest amount
of weight in the left portion of the domain. This occurs because the 6th
order expansion provides the highest fidelity prediction in
![(0.03, 0.1)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%280.03%2C%200.1%29 "(0.03, 0.1)")
compared to the lower order expansions. In other words, it has the
highest precision within this subregion. Since precision weighting is
used to calibrate the prior mean of the terminal node parameters, the
6th order expansion then receives higher weight apriori in this
subregion.

#### Results

<img src="EFT-Model-Mixing-Using-BART_files/figure-gfm/Ex3 Weights-1.png" style="display: block; margin: auto;" />

#### Code

``` r
#------------------------------------------------
# Plot the weight functions 
#------------------------------------------------
# Plot the weights from the non-informative prior
w1 = plot_wts_gg2(fitw1, ex3_data$x_test, y_lim = c(-0.05, 1.05), title = 'Non-Informative Prior', 
                  colors = color_list, line_type_list = lty_list, gray_scale = FALSE)
# Plot the weights from the informative prior
w2 = plot_wts_gg2(fitw2, ex3_data$x_test, y_lim = c(-0.05, 1.05), title = 'Informative Prior', 
                  colors = color_list, line_type_list = lty_list, gray_scale = FALSE)
# Resize text elements
w1 = w1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
w2 = w2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots
legend_w1 = g_legend(w1)
grid.arrange(arrangeGrob(w1 + theme(legend.position = "none"),
                         w2 + theme(legend.position = "none"),
                         nrow = 1),nrow=2, heights = c(10,1), legend = legend_w1,
                        top = textGrob("Posterior Weight Functions", gp = gpar(fontsize = 16)))
```

## Tuning the priors:

This section outlines some general guidelines and observations regarding
a few noteable tuning parameters in the model. Future work will look to
provide a more systematic guidance for selecting these parameters.

- `ntree` - the number of trees:

  - General:
    - When using a lower amount of trees (1-10), each individual tree is
      more influential on the results.
    - A higher number of trees typically results in smoother estimates
      of the weight functions.
  - Non-informative Prior:
    - Selecting a lower number of trees (5-10) generally yields very
      smooth predictions with little uncertainty in areas of the domain
      where at least one model provides a high fidelity prediction of
      the true system.
    - A trade-off with a lower number of trees is that the estimated
      uncertainty in the other areas of the domain (where no local
      expert exists) may slightly change for different runs of the MCMC.
      This occurs because the influence of an individual tree will be
      more apparent when only a few are included in the sum-of-trees
      model.
    - Meanwhile, with a higher number of trees (20-30), splits typically
      occur throughout the entire domain. When we are nearly
      interpolating, the uncertainty between the training points will
      form the typical bubble-like band that we come to expect.
    - This sensitivity to how the trees split was the reason for pursing
      the informative prior.
  - Informative Prior:
    - Because the weights are better informed with the prior, the number
      of trees selected is less influential on the results compared to
      the non-informative prior.

- `k`: – used in the prior variance of the terminal node parameters.

  - The value of `k` controls the flexibility of the model weights.
    Increasing `k` decreases the variability in the weights.
  - Larger `k` implies the prior is more influential in the posterior of
    the weight functions.

- `overallnu` – shape parameter in variance prior.

  - Increasing this value decreases the spread of the prior and
    concentrates its mass closer to the mean and mode of the
    distribution.
  - Typical values range from 3 to 10.

- `overallsd` – used to calibrate the scale parameter in the variance
  prior.

  - This value controls the range of values that the variance prior will
    be concentrated around.
  - Higher values of `overallsd` will shift the distribution to the
    right.

- `minnumbot` - the minimum number of observations per node:

  - This can influence the mean predictions of the system and the weight
    functions along with the associated variability.
  - Increasing this value can help reduce issues with overfitting.
  - When mixing EFTs that diverge rapidly (such as the 4th order strong
    coupling expansion), it is recommended to set this value to be
    greater than one.

<!-- * **Non-informative prior:** -->
<!--   -   Selecting a lower number of trees (5-10) generally yields very smooth predictions with little uncertainty in areas of the domain where at least one model provides a high fidelity prediction of the true system.  -->
<!--   - A trade-off with a lower number of trees is that the estimated uncertainty in the other areas of the domain (where no local expert exists) may slightly change for different runs of the MCMC. This occurs because the influence of an individual tree will be more apparent when only a few are included in the sum-of-trees model.  -->
<!--   - Meanwhile, with a higher number of trees (20-30), splits typically occur throughout the entire domain.  When we are nearly interpolating, the uncertainty between the training points will form the typical bubble-like band that we come to expect.    -->
<!--   - One positive from using more trees is that the estimated uncertainty in the intermediate range is more stable and hardly changes for different MCMC runs with the same settings of the tuning parameters.   -->
<!--   - This variability was the reason for pursuing an informative prior when mixing EFTs.  -->
<!--   - Note, results are less sensitive when considering models with mean predictions that do not diverge as rapidly as the ones in this motivating example. -->
<!-- * **Informative prior:**   -->
<!--   - When the precision weighting scheme (used to calibrate the prior mean), one can select higher values of $k$ (beyond 5) to further shrink the weight values towards the prior mean.  -->
<!--   - Typically, a higher amount of trees results in smoother uncertainty bands that appear more like bubbles rather than the sharp bands shown in the above examples (specifically Example 1). -->
<!--   - The estimates of the uncertainty are very consistent across MCMC runs. -->

## References

Honda, M. (2014), “On perturbation theory improved by strong coupling
expansion”, Journal of High Energy Physics 2014(12), 1–44.

Melendez, J. A., Furnstahl, R. J., Phillips, D. R., Pratola, M. T. and
Wesolowski, S. (2019), “Quantifying correlated truncation errors in
effective field theory”, Physical Review C 100(4).

Semposki, A., Furnstahl, R. and Phillips, D. (2022), “Uncertainties
here, there, and everywhere: interpolating between small-and large-g
expansions using Bayesian model mixing”, arXiv preprint
arXiv:2206.04116.
