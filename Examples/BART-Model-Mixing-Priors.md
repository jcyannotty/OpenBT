BART Model Mixing Priors
================
John Yannotty
11-18-2022

<style type="text/css">
  body{
  line-height: 2
  }
</style>
## Objective

This document serves as an online supplement to the article “Model
Mixing Using Bayesian Additive Regression Trees” by Yannotty et
al. (2022). The goal of this document is to illustrate the difference
between the proposed informative and non-informative priors for the
BART-based model.

## Introduction

Assume
![Y_1,\ldots,Y_n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y_1%2C%5Cldots%2CY_n "Y_1,\ldots,Y_n")
are independent and identically distributed (iid) random variables
originating from the true and underlying data generating mechanism

![Y_i = f\_\dagger(x_i) + \epsilon_i \quad \text{where} \\;\\; \epsilon_i \stackrel{iid}{\sim}N(0,\sigma^2)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y_i%20%3D%20f_%5Cdagger%28x_i%29%20%2B%20%5Cepsilon_i%20%5Cquad%20%5Ctext%7Bwhere%7D%20%5C%3B%5C%3B%20%5Cepsilon_i%20%5Cstackrel%7Biid%7D%7B%5Csim%7DN%280%2C%5Csigma%5E2%29 "Y_i = f_\dagger(x_i) + \epsilon_i \quad \text{where} \;\; \epsilon_i \stackrel{iid}{\sim}N(0,\sigma^2)")

and
![f\_\dagger(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_%5Cdagger%28x_i%29 "f_\dagger(x_i)")
is the true and unknown physical system evaluated at input
![x_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i "x_i").
Consider a set of models
![f_1(x_i),\ldots,f_K(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_1%28x_i%29%2C%5Cldots%2Cf_K%28x_i%29 "f_1(x_i),\ldots,f_K(x_i)"),
each of which are designed to explain the physical system across some
region of the domain. Assuming each model has been fit, the mean
predictions at inputs
![x_1,\ldots,x_n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_1%2C%5Cldots%2Cx_n "x_1,\ldots,x_n")
are given by
![\hat{f_1}(x_i),\ldots,\hat{f_K}(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bf_1%7D%28x_i%29%2C%5Cldots%2C%5Chat%7Bf_K%7D%28x_i%29 "\hat{f_1}(x_i),\ldots,\hat{f_K}(x_i)")
for
![i = 1,\ldots,n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i%20%3D%201%2C%5Cldots%2Cn "i = 1,\ldots,n").
Using the information from these mean predictions, the field data is
modeled as

![Y_i = \sum\_{l = 1}^Kw_l(x_i)\hat{f_l}(x_i) + \epsilon_i \quad \text{where} \\;\\; \epsilon_i \stackrel{iid}{\sim}N(0,\sigma^2).](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y_i%20%3D%20%5Csum_%7Bl%20%3D%201%7D%5EKw_l%28x_i%29%5Chat%7Bf_l%7D%28x_i%29%20%2B%20%5Cepsilon_i%20%5Cquad%20%5Ctext%7Bwhere%7D%20%5C%3B%5C%3B%20%5Cepsilon_i%20%5Cstackrel%7Biid%7D%7B%5Csim%7DN%280%2C%5Csigma%5E2%29. "Y_i = \sum_{l = 1}^Kw_l(x_i)\hat{f_l}(x_i) + \epsilon_i \quad \text{where} \;\; \epsilon_i \stackrel{iid}{\sim}N(0,\sigma^2).")

The weight functions
![\boldsymbol w(x_i) = \big(w_1(x_i),\ldots,w_K(x_i)\big)^\top](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%20w%28x_i%29%20%3D%20%5Cbig%28w_1%28x_i%29%2C%5Cldots%2Cw_K%28x_i%29%5Cbig%29%5E%5Ctop "\boldsymbol w(x_i) = \big(w_1(x_i),\ldots,w_K(x_i)\big)^\top")
are modeled using a sum-of-trees where
![T_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_j "T_j")
is the jth tree and
![M_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;M_j "M_j")
is the corresponding set of terminal node parameters. The output of each
tree can be expressed as a piecewise constant function

![\boldsymbol g(x_i, T_j,M_j) = \sum\_{p = 1}^{P_j} \boldsymbol\mu\_{pj} \boldsymbol 1\_{x_i \in \eta\_{pj}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%20g%28x_i%2C%20T_j%2CM_j%29%20%3D%20%5Csum_%7Bp%20%3D%201%7D%5E%7BP_j%7D%20%5Cboldsymbol%5Cmu_%7Bpj%7D%20%5Cboldsymbol%201_%7Bx_i%20%5Cin%20%5Ceta_%7Bpj%7D%7D "\boldsymbol g(x_i, T_j,M_j) = \sum_{p = 1}^{P_j} \boldsymbol\mu_{pj} \boldsymbol 1_{x_i \in \eta_{pj}}")

where
![P_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;P_j "P_j")
is the number of terminal nodes in
![T_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_j "T_j"),
![\eta\_{pj}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ceta_%7Bpj%7D "\eta_{pj}")
is the pth terminal node in
![T_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_j "T_j"),
![\boldsymbol\mu\_{pj} \in \mathbb{R}^K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cmu_%7Bpj%7D%20%5Cin%20%5Cmathbb%7BR%7D%5EK "\boldsymbol\mu_{pj} \in \mathbb{R}^K")
is the terminal node parameter associated with this node, and
![\boldsymbol 1\_{x_i \in \eta\_{pj}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%201_%7Bx_i%20%5Cin%20%5Ceta_%7Bpj%7D%7D "\boldsymbol 1_{x_i \in \eta_{pj}}")
is the indicator function that
![x_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i "x_i")
lies within the hyper-rectangle defined by
![\eta\_{pj}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ceta_%7Bpj%7D "\eta_{pj}").

The BART model mixing framework has the option to specify either an
informative or non-informative prior for the terminal node parameters
depending on the available information in the model set. The current
implementation of the informative prior is tailored to Effective Field
Theories (EFTs), which are techniques used to model physical systems of
interest. This method is described in detail by Yannotty et al. (2022).

In general, consider the jth tree,
![T_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_j "T_j")
with terminal nodes
![\eta\_{1j},\ldots,\eta\_{Pj}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ceta_%7B1j%7D%2C%5Cldots%2C%5Ceta_%7BPj%7D "\eta_{1j},\ldots,\eta_{Pj}")
(assuming
![T_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_j "T_j")
has
![P](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;P "P")
terminal nodes). Each terminal node is associated with a parameter
![\boldsymbol\mu\_{pj} \in \mathbb{R}^K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cmu_%7Bpj%7D%20%5Cin%20%5Cmathbb%7BR%7D%5EK "\boldsymbol\mu_{pj} \in \mathbb{R}^K").
This parameter is the contribution to the weight functions
![w_1(x),\ldots,w_K(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_1%28x%29%2C%5Cldots%2Cw_K%28x%29 "w_1(x),\ldots,w_K(x)")
from the jth Tree for those observations that are assigned to the
hyper-rectangle defined by
![\eta\_{pj}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ceta_%7Bpj%7D "\eta_{pj}")
where
![p = 1,\ldots,P](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;p%20%3D%201%2C%5Cldots%2CP "p = 1,\ldots,P").
In general, the parameter is assigned the prior

![\boldsymbol\mu\_{pj}\mid T_j \stackrel{ind}{\sim}N_K(\boldsymbol\beta\_{pj}, \tau^2 I_K)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cmu_%7Bpj%7D%5Cmid%20T_j%20%5Cstackrel%7Bind%7D%7B%5Csim%7DN_K%28%5Cboldsymbol%5Cbeta_%7Bpj%7D%2C%20%5Ctau%5E2%20I_K%29 "\boldsymbol\mu_{pj}\mid T_j \stackrel{ind}{\sim}N_K(\boldsymbol\beta_{pj}, \tau^2 I_K)")

where
![\boldsymbol\beta\_{pj} = (\beta\_{pj1},\ldots,\beta\_{pjK})^\top \in \mathbb{R}^K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cbeta_%7Bpj%7D%20%3D%20%28%5Cbeta_%7Bpj1%7D%2C%5Cldots%2C%5Cbeta_%7BpjK%7D%29%5E%5Ctop%20%5Cin%20%5Cmathbb%7BR%7D%5EK "\boldsymbol\beta_{pj} = (\beta_{pj1},\ldots,\beta_{pjK})^\top \in \mathbb{R}^K")
and
![I_K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;I_K "I_K")
is the
![\small{K\times K}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7BK%5Ctimes%20K%7D "\small{K\times K}")
identity matrix. The diagonal covariance structure implies that the
![K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;K "K")
components of
![\boldsymbol\mu\_{pj}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cmu_%7Bpj%7D "\boldsymbol\mu_{pj}")
are independent conditional on the tree. Furthermore, for fixed
![x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x "x"),
this implies the weight functions
![w_1(x),\ldots,w_K(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_1%28x%29%2C%5Cldots%2Cw_K%28x%29 "w_1(x),\ldots,w_K(x)")
are also conditionally independent.

The two different types of priors are illustrated throughout this
article. **See Yannotty et al. (2022) for a detailed discussion on the
calibration of the hyperparameters
![\boldsymbol\beta\_{pj}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cbeta_%7Bpj%7D "\boldsymbol\beta_{pj}")
and
![\tau^2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau%5E2 "\tau^2")
with either prior.** Before discussing these details, we briefly
summarize our motivating example involving EFTs.

## Motivating Example:

Assume 20 training points
![Y_1,\ldots,Y\_{20}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y_1%2C%5Cldots%2CY_%7B20%7D "Y_1,\ldots,Y_{20}")
are generated at inputs
![x_1,\ldots,x\_{20}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_1%2C%5Cldots%2Cx_%7B20%7D "x_1,\ldots,x_{20}")
according to

Consider a model set containing the 2nd order weak coupling expansion,
![f_s^{(2)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_s%5E%7B%282%29%7D%28x%29 "f_s^{(2)}(x)"),
and the 4th order strong coupling expansion,
![f_l^{(4)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_l%5E%7B%284%29%7D%28x%29 "f_l^{(4)}(x)").
Given this set of EFTs, the data is modeled as

![Y_i = w_1(x_i)\hat{f_s}^{(2)}(x_i) + w_2(x_i)\hat{f_l}^{(4)}(x_i) + \epsilon_i, \quad \text{where} \\;\\; \epsilon_i \stackrel{iid}{\sim}N(0,\sigma^2)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y_i%20%3D%20w_1%28x_i%29%5Chat%7Bf_s%7D%5E%7B%282%29%7D%28x_i%29%20%2B%20w_2%28x_i%29%5Chat%7Bf_l%7D%5E%7B%284%29%7D%28x_i%29%20%2B%20%5Cepsilon_i%2C%20%5Cquad%20%5Ctext%7Bwhere%7D%20%5C%3B%5C%3B%20%5Cepsilon_i%20%5Cstackrel%7Biid%7D%7B%5Csim%7DN%280%2C%5Csigma%5E2%29 "Y_i = w_1(x_i)\hat{f_s}^{(2)}(x_i) + w_2(x_i)\hat{f_l}^{(4)}(x_i) + \epsilon_i, \quad \text{where} \;\; \epsilon_i \stackrel{iid}{\sim}N(0,\sigma^2)")

and
![\hat{f_s}^{(2)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bf_s%7D%5E%7B%282%29%7D%28x%29 "\hat{f_s}^{(2)}(x)")
and
![\hat{f_l}^{(4)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bf_l%7D%5E%7B%284%29%7D%28x%29 "\hat{f_l}^{(4)}(x)")
are the mean predictions from each EFT. The mean predictions are shown
below in Figure 1. Clearly, each model provides high fidelity
predictions in one region, yet diverges elsewhere.

<img src="BART-Model-Mixing-Priors_files/figure-gfm/Plot Expansion-1.png" style="display: block; margin: auto;" />

Each EFT has an associated stochastic truncation error,
![\delta(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cdelta%28x%29 "\delta(x)"),
which indicates its level of accuracy for predicting the true system. In
areas where
![\delta(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cdelta%28x%29 "\delta(x)")
has a high variance, the corresponding EFT provides a poor approximation
of the physical system. For simplicity, we use the pointwise error
approximation described by Semposki et al. (2022) to fit the two EFTs in
this example. The weak and strong coupling expansions under
consideration are shown in Figure 2 with their associated 95% pointwise
confidence intervals. Clearly, when an EFT provides a poor approximation
of the system its mean prediction diverges and its corresponding
variance from the truncation error is high. Note, the mean predictions,
![\hat{f_1}(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bf_1%7D%28x_i%29 "\hat{f_1}(x_i)")
and
![\hat{f_2}(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bf_2%7D%28x_i%29 "\hat{f_2}(x_i)"),
at the training points
![x_1,\ldots,x\_{20}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_1%2C%5Cldots%2Cx_%7B20%7D "x_1,\ldots,x_{20}")
are represented as points on the respective curves below.

  

<img src="BART-Model-Mixing-Priors_files/figure-gfm/Plot Expansion and Delta-1.png" style="display: block; margin: auto;" />

## Non-Informative Prior:

### Summary:

**Objective:**

- The non-informative prior is designed to work for any model mixing
  problem where the mean predictions,
  ![\hat{f_1}(x),\ldots,\hat{f_K}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bf_1%7D%28x%29%2C%5Cldots%2C%5Chat%7Bf_K%7D%28x%29 "\hat{f_1}(x),\ldots,\hat{f_K}(x)"),
  from each model are obtainable. This method does not require any
  additional information from the model set.

- The hyperparameters in each terminal node model are calibrated by
  considering the entire sum-of-trees model.

**Model:**

- ![\boldsymbol\mu\_{pj} \mid T_j \stackrel{iid}{\sim}N_K\big(\boldsymbol\beta, \tau^2 I_K\big)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cmu_%7Bpj%7D%20%5Cmid%20T_j%20%5Cstackrel%7Biid%7D%7B%5Csim%7DN_K%5Cbig%28%5Cboldsymbol%5Cbeta%2C%20%5Ctau%5E2%20I_K%5Cbig%29 "\boldsymbol\mu_{pj} \mid T_j \stackrel{iid}{\sim}N_K\big(\boldsymbol\beta, \tau^2 I_K\big)")
  where
  ![\boldsymbol\mu\_{pj} = (\mu\_{pj1},\ldots,\mu\_{pjK})^\top](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cmu_%7Bpj%7D%20%3D%20%28%5Cmu_%7Bpj1%7D%2C%5Cldots%2C%5Cmu_%7BpjK%7D%29%5E%5Ctop "\boldsymbol\mu_{pj} = (\mu_{pj1},\ldots,\mu_{pjK})^\top")
  and
  ![\boldsymbol\beta= (\beta\_{1},\ldots,\beta\_{K})^\top](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cbeta%3D%20%28%5Cbeta_%7B1%7D%2C%5Cldots%2C%5Cbeta_%7BK%7D%29%5E%5Ctop "\boldsymbol\beta= (\beta_{1},\ldots,\beta_{K})^\top").

- The prior induced by the indepdent and identically distributed (iid)
  assumption and the sum-of-trees implies
  ![w_l(x_i)\mid T_j \stackrel{iid}{\sim}N(m\beta_l, m\tau^2)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_l%28x_i%29%5Cmid%20T_j%20%5Cstackrel%7Biid%7D%7B%5Csim%7DN%28m%5Cbeta_l%2C%20m%5Ctau%5E2%29 "w_l(x_i)\mid T_j \stackrel{iid}{\sim}N(m\beta_l, m\tau^2)")
  for
  ![l = 1,\ldots,K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%20%3D%201%2C%5Cldots%2CK "l = 1,\ldots,K").

- The hyperparameters are calibrated as
  ![\beta_l = \frac{1}{2m}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_l%20%3D%20%5Cfrac%7B1%7D%7B2m%7D "\beta_l = \frac{1}{2m}")
  and
  ![\tau = \frac{1}{2k\sqrt{m}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau%20%3D%20%5Cfrac%7B1%7D%7B2k%5Csqrt%7Bm%7D%7D "\tau = \frac{1}{2k\sqrt{m}}")
  for
  ![l = 1,\ldots,K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%20%3D%201%2C%5Cldots%2CK "l = 1,\ldots,K")
  and tuning parameter
  ![k](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;k "k").

### Prior Predictions with a Non-Informative Prior

With the non-informative prior,
![w_l(x_i)\mid T_j \stackrel{iid}{\sim}N(\frac{1}{2}, m\tau^2)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_l%28x_i%29%5Cmid%20T_j%20%5Cstackrel%7Biid%7D%7B%5Csim%7DN%28%5Cfrac%7B1%7D%7B2%7D%2C%20m%5Ctau%5E2%29 "w_l(x_i)\mid T_j \stackrel{iid}{\sim}N(\frac{1}{2}, m\tau^2)")
for
![l = 1,\ldots,K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%20%3D%201%2C%5Cldots%2CK "l = 1,\ldots,K")
at any fixed
![x_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i "x_i")
where
![\tau = \frac{1}{2k\sqrt{m}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau%20%3D%20%5Cfrac%7B1%7D%7B2k%5Csqrt%7Bm%7D%7D "\tau = \frac{1}{2k\sqrt{m}}").
As a starting point, we can observe the pointwise prior predictions of
the weight functions and the induced mean prediction of the physical
system. Both of these features are shown below for
![k = 2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;k%20%3D%202 "k = 2")
and a 10-tree model
(i.e. ![m = 10](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;m%20%3D%2010 "m = 10")).
Clearly, the weight functions lie on top of each other with a 95%
confidence interval ranging from (0, 1). Meanwhile, the prior prediction
of the system is merely an average of the two EFTs. Hence, under the
non-informative prior, the observational data is forced to play a
prominent role in the posterior prediction in order to adequately
recover the true system.

  

<img src="BART-Model-Mixing-Priors_files/figure-gfm/noninform pointwise prior-1.png" style="display: block; margin: auto;" />

  

## Informative Prior:

### Summary:

**Objective:**:

- The informative prior allows the user to directly incorporate
  information regarding each models localized predictive performance
  into the prior of the terminal node parameters.

- This prior considers the effect of an individual parameter/tree on the
  overall weight.

- The current implementation is tailored to EFT models by leveraging the
  information available in the truncation errors.

**Model:**

- ![\boldsymbol\mu\_{pj} \mid T_j \stackrel{ind}{\sim}N_2\big(\boldsymbol\beta\_{pj}, \tau^2 I_2\big)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cmu_%7Bpj%7D%20%5Cmid%20T_j%20%5Cstackrel%7Bind%7D%7B%5Csim%7DN_2%5Cbig%28%5Cboldsymbol%5Cbeta_%7Bpj%7D%2C%20%5Ctau%5E2%20I_2%5Cbig%29 "\boldsymbol\mu_{pj} \mid T_j \stackrel{ind}{\sim}N_2\big(\boldsymbol\beta_{pj}, \tau^2 I_2\big)")
  where
  ![\boldsymbol\mu\_{pj} = (\mu\_{pj1},\ldots,\mu\_{pjK})^\top](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cmu_%7Bpj%7D%20%3D%20%28%5Cmu_%7Bpj1%7D%2C%5Cldots%2C%5Cmu_%7BpjK%7D%29%5E%5Ctop "\boldsymbol\mu_{pj} = (\mu_{pj1},\ldots,\mu_{pjK})^\top")
  and
  ![\boldsymbol\beta\_{pj} = (\beta\_{pj1},\ldots,\beta\_{pjK})^\top](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%5Cbeta_%7Bpj%7D%20%3D%20%28%5Cbeta_%7Bpj1%7D%2C%5Cldots%2C%5Cbeta_%7BpjK%7D%29%5E%5Ctop "\boldsymbol\beta_{pj} = (\beta_{pj1},\ldots,\beta_{pjK})^\top").

- Allow
  ![\beta\_{pjl}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_%7Bpjl%7D "\beta_{pjl}")
  to change based on the partitions induced by the tree, where
  ![l=1,\ldots,K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%3D1%2C%5Cldots%2CK "l=1,\ldots,K").

- Fix
  ![\tau = \frac{1}{2km}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau%20%3D%20%5Cfrac%7B1%7D%7B2km%7D "\tau = \frac{1}{2km}")
  so that each tree explains
  ![1/m^{th}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%2Fm%5E%7Bth%7D "1/m^{th}")
  of the variation in the weight functions.

  

### Illustrating the Prior Calibration

The follow steps demonstrate how the terminal node hyperparameters for a
given tree are calibrated during the MCMC. For this example, assume a
10-tree model
(![\small{m =10}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7Bm%20%3D10%7D "\small{m =10}"))
with the tuning parameter
![\small{k = 2}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7Bk%20%3D%202%7D "\small{k = 2}").
Specifically, we will focus on calibrating the terminal node parameters
for the first tree in the ensemble,
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1").
The same steps can be taken to calibrate the priors of the terminal node
parameters in the remaining nine trees.

  

#### **Step 1: Initial Pointwise Estimates of the Weights**

Each EFT is associated with a truncation error which indicates the
model’s localized predictive accuracy. The predictive accuracy of the
true system is inversely related to the variance of the truncation
error. This can easily be observed by looking at the results in Figures
1 and 2. We can use this information and a precision weighting scheme to
provide an initial guess for each model weight. In particular, define
![\beta_1(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_1%28x_i%29 "\beta_1(x_i)")
and
![\beta_2(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_2%28x_i%29 "\beta_2(x_i)")
as the initial guesses for
![w_1(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_1%28x_i%29 "w_1(x_i)")
and
![w_2(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_2%28x_i%29 "w_2(x_i)")
respectively. Each function can be defined by

![\beta_l(x_i) = \frac{1/v_l(x_i)}{1/v_1(x_i) + 1/v_2(x_i)}, \quad l = 1,2.](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_l%28x_i%29%20%3D%20%5Cfrac%7B1%2Fv_l%28x_i%29%7D%7B1%2Fv_1%28x_i%29%20%2B%201%2Fv_2%28x_i%29%7D%2C%20%5Cquad%20l%20%3D%201%2C2. "\beta_l(x_i) = \frac{1/v_l(x_i)}{1/v_1(x_i) + 1/v_2(x_i)}, \quad l = 1,2.")

This precision weighting scheme can be applied at each of the 20
training inputs
![x_1,\ldots,x\_{20}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_1%2C%5Cldots%2Cx_%7B20%7D "x_1,\ldots,x_{20}")
before the mixed-model is trained and the weight functions are learned.
Using this initial guess, we can interpret
![\beta_l(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_l%28x_i%29 "\beta_l(x_i)")
as the prior mean for its corresponding weight,
i.e. ![w_l(x_i) \stackrel{ind}{\sim}N\big(\beta_l(x_i), \tau^2\big)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_l%28x_i%29%20%5Cstackrel%7Bind%7D%7B%5Csim%7DN%5Cbig%28%5Cbeta_l%28x_i%29%2C%20%5Ctau%5E2%5Cbig%29 "w_l(x_i) \stackrel{ind}{\sim}N\big(\beta_l(x_i), \tau^2\big)")
for each
![i=1,\ldots,20](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i%3D1%2C%5Cldots%2C20 "i=1,\ldots,20")
and
![l=1,2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;l%3D1%2C2 "l=1,2").
The plots below illustrate the pointwise prior mean weights
![\beta_l(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_l%28x_i%29 "\beta_l(x_i)")
at each of the training points along with the pointwise prior
prediction, which is given by

![\hat{f\_\dagger}(x) = \beta_1(x)\hat{f_s}^{(2)}(x) + \beta_2(x)\hat{f_l}^{(4)}(x).](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bf_%5Cdagger%7D%28x%29%20%3D%20%5Cbeta_1%28x%29%5Chat%7Bf_s%7D%5E%7B%282%29%7D%28x%29%20%2B%20%5Cbeta_2%28x%29%5Chat%7Bf_l%7D%5E%7B%284%29%7D%28x%29. "\hat{f_\dagger}(x) = \beta_1(x)\hat{f_s}^{(2)}(x) + \beta_2(x)\hat{f_l}^{(4)}(x).")

Similar to the non-informative prior, these results are shown for
![k = 2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;k%20%3D%202 "k = 2")
and
![m = 10](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;m%20%3D%2010 "m = 10")
trees.

  

<img src="BART-Model-Mixing-Priors_files/figure-gfm/inform pointwise prior-1.png" style="display: block; margin: auto;" />

From this, we see the pointwise prior precision weights suggest the
weight functions
![w_1(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_1%28x_i%29 "w_1(x_i)")
and
![w_2(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;w_2%28x_i%29 "w_2(x_i)")
should take a sigmoid-like shape in order to properly predict the
system. Even though the pointwise prior prediction leaves much to be
desired in the intermediate range of the domain, it is a clear
improvement compared to the initial prediction under the non-informative
prior. The precision based weights appear to be a useful source of
information and at least can be used to guide the learning process with
the BART-based mixing. Rather than starting with a mean of
![1/2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%2F2 "1/2"),
the weights start at more realistic values and with a useful shape that
can be refined with the observational data. Thus, the posterior
predictions can borrow information from the prior and the observational
data to adequately recover the true system. This is not the case when
using a non-informative prior as described in the previous section, as
the solution is very dependent on the observational data.

  

#### **Step 2: Condition on a Tree Structure**

Consider calibrating the terminal node parameter priors for the first
tree in the ensemble,
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1").
For each node, the prior is defined conditional on the tree structure.
Thus, for this demonstration, assume
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1")
has the structure shown below:

<img src="BART-Model-Mixing-Priors_files/figure-gfm/tree1-1.png" style="display: block; margin: auto;" />

This is a shallow tree with three terminal nodes, hence the 20 training
points are divided into three partitions. These partitions of the input
space are explicitly shown by the green vertical lines in the right
panel of Figure 5.  
Based on the tree and the induced partitions, we can make the following
observations. Recall, this example uses 10 different trees in the
sum-of-trees model, thus each tree is expected to explain
![1/10th](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%2F10th "1/10th")
of the variation in the total weight functions. As a result, we expect
the terminal node parameters for one tree to prefer the interval
![\[0, 0.1\]](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5B0%2C%200.1%5D "[0, 0.1]").

Comments:

1.  Partition 1
    (![x_i \< 0.15](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i%20%3C%200.15 "x_i < 0.15"),
    ![\eta\_{11}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ceta_%7B11%7D "\eta_{11}")):
    The second order weak coupling expansion,
    ![f_s^{(2)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_s%5E%7B%282%29%7D%28x%29 "f_s^{(2)}(x)"),
    aligns very closely to the true system while the fourth order strong
    coupling expansion,
    ![f_l^{(4)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_l%5E%7B%284%29%7D%28x%29 "f_l^{(4)}(x)"),
    diverges and is far from the true system. From Figure 2, this
    relationship can also be discerned from the truncation errors. In
    terms of the terminal node parameters (i.e. this tree’s contribution
    to each model weight) we would expect
    ![\mu\_{111}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7B111%7D "\mu_{111}")
    to take values close to 1 and
    ![\mu\_{112}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7B112%7D "\mu_{112}")
    to take values close to 0 for any
    ![x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x "x")
    in this subregion.

2.  Partition 2
    (![0.15 \le x_i \< 0.35](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;0.15%20%5Cle%20x_i%20%3C%200.35 "0.15 \le x_i < 0.35"),
    ![\eta\_{21}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ceta_%7B21%7D "\eta_{21}")):
    The true system lies between both EFTs across this region. For
    ![x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x "x")
    closer to 0.15,
    ![f_s^{(2)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_s%5E%7B%282%29%7D%28x%29 "f_s^{(2)}(x)")
    provides a more accurate prediction that
    ![f_l^{(4)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_l%5E%7B%284%29%7D%28x%29 "f_l^{(4)}(x)").
    The reverse behavior is observed for
    ![x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x "x")
    closer to 0.35. This relationship can also be inferred from the
    truncation errors. Thus, we expect
    ![\mu\_{211}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7B211%7D "\mu_{211}")
    and
    ![\mu\_{212}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7B212%7D "\mu_{212}")
    should take values towards the away from the extremes of
    ![0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;0 "0")
    and
    ![1/m](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%2Fm "1/m").

3.  Partition 3
    (![x_i \ge 0.35](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i%20%5Cge%200.35 "x_i \ge 0.35"),
    ![\eta\_{31}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ceta_%7B31%7D "\eta_{31}")):
    ![f_l^{(4)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_l%5E%7B%284%29%7D%28x%29 "f_l^{(4)}(x)")
    yields high fidelity predictions of the true system in this
    subregion, while
    ![f_s^{(2)}(x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;f_s%5E%7B%282%29%7D%28x%29 "f_s^{(2)}(x)")
    slowly diverges. Thus, we can expect
    ![\mu\_{311}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7B311%7D "\mu_{311}")
    to take low values close to
    ![0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;0 "0")
    and
    ![\mu\_{312}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7B312%7D "\mu_{312}")
    to take values close to
    ![0.10](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;0.10 "0.10").

In the informative prior, the goal is to build this intuition into the
prior mean of the terminal node parameters. This can be accomplished by
utilizing the prior mean weight functions
![\beta_1(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_1%28x_i%29 "\beta_1(x_i)")
and
![\beta_2(x_i)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_2%28x_i%29 "\beta_2(x_i)")
which are based on the variances of the truncation errors.

  

#### **Step 3: Calibrating the Prior Mean**

Conditional on
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1")
from Step 2, we see there are three prior means which need to be
specified:
![\small{\boldsymbol\beta\_{11}, \boldsymbol\beta\_{21}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Cboldsymbol%5Cbeta_%7B11%7D%2C%20%5Cboldsymbol%5Cbeta_%7B21%7D%7D "\small{\boldsymbol\beta_{11}, \boldsymbol\beta_{21}}")
and
![\small{\boldsymbol\beta\_{31}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Cboldsymbol%5Cbeta_%7B31%7D%7D "\small{\boldsymbol\beta_{31}}").
In this problem, each mean vector
![\small{\boldsymbol\beta\_{31}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Cboldsymbol%5Cbeta_%7B31%7D%7D "\small{\boldsymbol\beta_{31}}")
is a two-dimensional vector with components
![\small{\beta\_{p11}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Cbeta_%7Bp11%7D%7D "\small{\beta_{p11}}")
and
![\small{\beta\_{p12}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Cbeta_%7Bp12%7D%7D "\small{\beta_{p12}}")
where
![\small{p = 1,2,3}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7Bp%20%3D%201%2C2%2C3%7D "\small{p = 1,2,3}").

Now consider one of the partitions induced by the tree. For simplicity,
focus on the first partition defined by
![x_i \< 0.15](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i%20%3C%200.15 "x_i < 0.15"),
which includes the first five training points
![(x_1, y_1),\ldots,(x_5, y_5)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%28x_1%2C%20y_1%29%2C%5Cldots%2C%28x_5%2C%20y_5%29 "(x_1, y_1),\ldots,(x_5, y_5)").
As described in Step 2, the weak coupling expansion should be given a
weight near one while the effect of the strong coupling expansion should
be shrunk towards zero. To express this idea mathematically, the prior
mean vector is set as:

![\beta\_{111} = \frac{1}{10}\sum\_{i = 1}^5\frac{\beta_1(x_i)}{5}  \quad \text{and} \quad \beta\_{112} = \frac{1}{10}\sum\_{i = 1}^5\frac{\beta_2(x_i)}{5}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_%7B111%7D%20%3D%20%5Cfrac%7B1%7D%7B10%7D%5Csum_%7Bi%20%3D%201%7D%5E5%5Cfrac%7B%5Cbeta_1%28x_i%29%7D%7B5%7D%20%20%5Cquad%20%5Ctext%7Band%7D%20%5Cquad%20%5Cbeta_%7B112%7D%20%3D%20%5Cfrac%7B1%7D%7B10%7D%5Csum_%7Bi%20%3D%201%7D%5E5%5Cfrac%7B%5Cbeta_2%28x_i%29%7D%7B5%7D "\beta_{111} = \frac{1}{10}\sum_{i = 1}^5\frac{\beta_1(x_i)}{5}  \quad \text{and} \quad \beta_{112} = \frac{1}{10}\sum_{i = 1}^5\frac{\beta_2(x_i)}{5}")

This calibration sets the prior means of the terminal node parameters
![\mu\_{111}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7B111%7D "\mu_{111}")
and
![\mu\_{112}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7B112%7D "\mu_{112}")
to be the sample means of the pointwise precision based weights from the
EFTs scaled by the number of trees (10). The scaling of
![1/10](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%2F10 "1/10")
is used to ensure each tree explains approximately
![1/10th](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%2F10th "1/10th")
of the variation in the model weights.

The same procedure is used to calibrate the remaining prior mean vectors
![\small{\boldsymbol\beta\_{21}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Cboldsymbol%5Cbeta_%7B21%7D%7D "\small{\boldsymbol\beta_{21}}")
and
![\small{\boldsymbol\beta\_{31}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csmall%7B%5Cboldsymbol%5Cbeta_%7B31%7D%7D "\small{\boldsymbol\beta_{31}}").
The calibrated values are shown below. Note, results have been rounded.

#### 

<!-- ##### Terminal Node Means -->
<!-- ```{r} -->
<!-- beta_df = data.frame(beta1,beta2,beta3) -->
<!-- beta_df = round(beta_df, 4) -->
<!-- rownames(beta_df) = c('$\\beta_{1}$ Component','$\\beta_{2}$ Component') -->
<!-- colnames(beta_df) = c('$\\eta_{11}$','$\\eta_{21}$','$\\eta_{21}$') -->
<!-- beta_df %>% kable() %>%  -->
<!--   kable_paper(full_width = F, position = 'center') %>%  -->
<!--   add_header_above(c("Prior Means by Terminal Node in $T_1$"=4)) -->
<!-- ``` -->
<!-- ##### Pointwise Data by Node -->
<!-- ```{r} -->
<!-- pw_df = data.frame(Input = ex1_data$x_train, betax) -->
<!-- colnames(pw_df)[1:3] = c('$x_i$','$\\beta_{1}(x)$','$\\beta_{2}(x)$') -->
<!-- pw_df = round(pw_df,6) -->
<!-- pw_df[,2] = ifelse(pw_df[,2] == 1, "1.0", as.character(format(pw_df[,2],scientific = F))) -->
<!-- pw_df[,3] = ifelse(pw_df[,3] == 1, "1.0", as.character(format(pw_df[,3],scientific = F))) -->
<!-- pw_df[,2] = ifelse(as.numeric(pw_df[,2]) == 0, "0", as.character(format(pw_df[,2],scientific = F))) -->
<!-- pw_df[,3] = ifelse(as.numeric(pw_df[,3]) == 0, "0", as.character(format(pw_df[,3],scientific = F))) -->
<!-- pw_df %>% kable(escape = F, align = 'c') %>%  -->
<!--   kable_paper(full_width = F) %>%  -->
<!--   add_header_above(c("Pointwise Prior Mean Values by Terminal Node in $T_1$"=3)) %>%  -->
<!--   pack_rows("Node 1", 1, 5) %>% -->
<!--   pack_rows("Node 2", 6, 13) %>% -->
<!--   pack_rows("Node 3", 14, 20) %>%  -->
<!--   scroll_box(height = "400px") -->
<!-- ``` -->

  

#### **Step 4: Calibrating the Prior Standard Deviation**

The standard deviation,
![\tau](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau "\tau")
does not change based on the tree partitions. Thus, it is selected so
that the confidence interval width for any terminal node parameter
![\mu\_{pjl}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7Bpjl%7D "\mu_{pjl}")
is
![1/m](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%2Fm "1/m").
Assuming a symmetric confidence interval, this idea is expressed as:

![\beta\_{pjl} - \frac{1}{2m} = \beta\_{pjl} - k\tau  \quad \text{and} \quad \beta\_{pjl} + \frac{1}{2m} = \beta\_{pjl} + k\tau](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta_%7Bpjl%7D%20-%20%5Cfrac%7B1%7D%7B2m%7D%20%3D%20%5Cbeta_%7Bpjl%7D%20-%20k%5Ctau%20%20%5Cquad%20%5Ctext%7Band%7D%20%5Cquad%20%5Cbeta_%7Bpjl%7D%20%2B%20%5Cfrac%7B1%7D%7B2m%7D%20%3D%20%5Cbeta_%7Bpjl%7D%20%2B%20k%5Ctau "\beta_{pjl} - \frac{1}{2m} = \beta_{pjl} - k\tau  \quad \text{and} \quad \beta_{pjl} + \frac{1}{2m} = \beta_{pjl} + k\tau")

Solving either equation implies
![\tau = \frac{1}{2km}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau%20%3D%20%5Cfrac%7B1%7D%7B2km%7D "\tau = \frac{1}{2km}").
With
![k = 2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;k%20%3D%202 "k = 2")
and
![m = 10](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;m%20%3D%2010 "m = 10"),
we have
![\tau = \frac{1}{40} = 0.025](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau%20%3D%20%5Cfrac%7B1%7D%7B40%7D%20%3D%200.025 "\tau = \frac{1}{40} = 0.025").

  

### Plotting the Prior

The joint prior distribution of
![\mu\_{p11}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7Bp11%7D "\mu_{p11}")
and
![\mu\_{p12}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7Bp12%7D "\mu_{p12}")
for each
![p = 1,2,3](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;p%20%3D%201%2C2%2C3 "p = 1,2,3")
are shown below in Figure 6. Each plot illustrates a bivariate normal
distribution with standard deviation
![\tau = 0.025](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau%20%3D%200.025 "\tau = 0.025").
The contours in each plot represent the number of standard deviations
from the mean. The clear difference between each prior is the location
of the center of the distribution. In the first node, the prior is
centered around
![(0.1, 0)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%280.1%2C%200%29 "(0.1, 0)")
which indicates the prior belief that second order expansion is closely
aligned with the true system in this region while the fourth order
expansion diverges. The joint distribution in the second panel is
centered around
![(0.076, 0.024)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%280.076%2C%200.024%29 "(0.076, 0.024)")
which indicates the second order expansion has the better overall
performance in this middle subregion compared to the fourth order
expansion. Finally, the third distribution is centered around
![(0, 0.10)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%280%2C%200.10%29 "(0, 0.10)")
which indicates the fourth order expansion is the appropriate model in
this region. In summary, it appears the prior distributions shown below
match our intuition about how the models should be mixed given the tree
structure defined by
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1").

  

<img src="BART-Model-Mixing-Priors_files/figure-gfm/tnode prior-1.png" style="display: block; margin: auto;" />

  

### Prior Predictions with an Informative Prior

Given the priors define in the steps above, now consider the prior
weight functions and mean predictions associated with
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1").
The prior mean predictions are given by

![\hat{f\_\dagger}(x_i) = \beta\_{p11}\hat{f_s}^{(2)}(x_i) + \beta\_{p12}\hat{f_l}^{(4)}(x_i) \quad \text{if} \\;\\; x_i \in \eta\_{pj}, \\;\\; p=1,2,3.](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bf_%5Cdagger%7D%28x_i%29%20%3D%20%5Cbeta_%7Bp11%7D%5Chat%7Bf_s%7D%5E%7B%282%29%7D%28x_i%29%20%2B%20%5Cbeta_%7Bp12%7D%5Chat%7Bf_l%7D%5E%7B%284%29%7D%28x_i%29%20%5Cquad%20%5Ctext%7Bif%7D%20%5C%3B%5C%3B%20x_i%20%5Cin%20%5Ceta_%7Bpj%7D%2C%20%5C%3B%5C%3B%20p%3D1%2C2%2C3. "\hat{f_\dagger}(x_i) = \beta_{p11}\hat{f_s}^{(2)}(x_i) + \beta_{p12}\hat{f_l}^{(4)}(x_i) \quad \text{if} \;\; x_i \in \eta_{pj}, \;\; p=1,2,3.")

Note, Figure 7 illustrates the contribution of
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1"),
as opposed to the result of the entire sum-of-trees. Thus, the y-axis
scale in Figure 7 is smaller than the plots displayed earlier. The prior
distributions of
![\mu\_{pj1}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7Bpj1%7D "\mu_{pj1}")
and
![\mu\_{pj2}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu_%7Bpj2%7D "\mu_{pj2}")
are shown as a function of
![x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x "x")
and exhibit the piecewise function defined by each tree. The right panel
shows the contribution to the overall mean prediction from
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1").
The prior mean prediction with the informative prior is shown in purple,
while the prior mean prediction from the non-informative prior is shown
in orange. It appears the mean prediction from
![T_1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_1 "T_1")
under the informative prior begins to embody the general shape of the
true function. This is a noticeable improvement than the result shown
from the non-informative prior where each terminal node parameter is
centered at
![1/2m = 0.05](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%2F2m%20%3D%200.05 "1/2m = 0.05").
This yields an average of the predictions from the individual EFTs and
clearly diverges in the rightmost portion of the domain.

  

<img src="BART-Model-Mixing-Priors_files/figure-gfm/inform tnode prior-1.png" style="display: block; margin: auto;" />
