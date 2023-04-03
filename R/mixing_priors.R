#-----------------------------------------------------
# Model Mixing Prior and Posterior Functions

#-----------------------------------------------------
# Terminal node priors
#-----------------------------------------------------
# Non-Informative prior
noninf_prior = function(num_models = 2,k=2,m=1){
  # Prior mean and std. deviation
  beta = rep(1/(2*m), num_models)
  tau = (1)/(2*k*sqrt(m))
  tau = rep(tau, num_models)
  
  out = list(beta = beta, tau = tau)
  return(out)  
}

# Informative EFT
eft_prior = function(std_matrix, k = 2, m = 1){
  # Get beta(x)
  prec_matrix = 1/dsd.train^2 #Precision Matrix
  betax_matrix = prec_matrix/rowSums(prec_matrix) 
  
  # Get constants tau and beta
  beta = (1/m)*apply(betax_matrix,2,mean)
  tau = 1/(2*k*m) #The factor of sqrt m appears out front because this is the pw prior for w(x) not mu
  tau = rep(tau, num_models)
  out = list(beta = beta, tau = tau)
  return(out) 
}

# Selection Prior -- discrete hierarchical component for beta
# -- sc = selected component
select_prior = function(num_models,sc,k=2, m=1){
  # Set beta and tau
  beta = rep(0,num_models)
  beta[sc] = 1/m
  tau = 1/(2*k*m)
  tau = rep(tau, num_models)
  out = list(beta = beta, tau = tau)
  return(out) 
}

# Soft BART Split
sbart_split = function(x,c,s,r=TRUE){
  x0 = (x-c)/s
  psi = 1/(1 + exp(-x0))
  if(r){
    prob = psi   
  }else{
    prob = 1 - psi
  }
  return(prob)
}

# SBART phi(x)
# -- path = vector of true/false, path[i] = true if move right at ith internal node on path
# -- cvec and xvec, the cutpoint and x value which determine the move
sbart_phix = function(path, xvec, cvec, s){
  log_phi = 0
  for(i in 1:length(path)){
    prb = sbart_split(xvec[i],cvec[i],s,r=path[i])
    log_phi = log_phi + log(prb)
  }
  phi = exp(log_phi)
  return(phi)
}

# Random Assignments
sbart_drawz = function(phix){
  B = ncol(phix)
  n = nrow(phix)
  z = apply(phix, 1, function(x) sample(1:B, size = 1, prob = x))
  z_ind = matrix(0,ncol = B, nrow = n)
  for(i in 1:n){
    z_ind[i,z[i]] = 1
  }
  return(z_ind)
}

#-----------------------------------------------------
# Prior prediction functions -- for all nodes
# -- tnode_list: list, each item is a vector of indexes corresponding to which obs are assigned to the node
# -- beta matrix, BxK, each row is the prior mean for bth tnode vector
# -- tau matrix, BxK, each row is the prior mean for bth tnode vector
#-----------------------------------------------------
prior_predict = function(f_matrix, tnode_list, beta_matrix, tau_matrix){
  # Setup
  n = nrow(f_matrix)
  predmean = rep(0,n)
  predstd = rep(0,n)
  B = length(tnode_list)
  # Get predictions by terminal node
  for(b in 1:B){
    h = tnode_list[[b]]
    predmean[h] = f_matrix[h,]%*%beta_matrix[b,]
    predstd[h] = sqrt(diag(f_matrix[h,]%*%diag(tau_matrix[b,]^2)%*%t(f_matrix[h,])))
  }
  outpred = list(predmean = predmean, predstd = predstd)
  return(outpred)
}

# Soft prior predict
soft_prior_predict = function(f_matrix, beta_matrix, tau_matrix, phi_matrix){
  n = nrow(f_matrix)
  predmean = rep(0,n)
  predstd = rep(0,n)
  B = nrow(beta_matrix)
  for(b in 1:B){
    predmean = predmean + phi_matrix[b,]*f_matrix%*%beta_matrix[b,]
    predstd = predstd + phi_matrix[b,]*sqrt(diag(f_matrix%*%diag(tau_matrix[b,]^2)%*%t(f_matrix)))
  }
  
  outpred = list(predmean = predmean, predstd = predstd)
  return(outpred)
}

#-----------------------------------------------------
# Terminal Node Posterior functions
#-----------------------------------------------------
# General
post_predict = function(f_matrix, tnode_list, beta_matrix, tau_matrix, y_vec, sigma){
  # Define terms
  K = ncol(f_matrix)
  n = nrow(f_matrix)
  B = length(tnode_list)
  predmean = predstd = rep(0,n)
  # Loop through tnodes
  for(j in 1:B){
    # Terms
    h = tnode_list[[j]]
    f = f_matrix[h,]
    y = y_vec[h] 
    prior_inv = diag(1/tau_matrix[j,]^2)
    
    # Mean and Variance
    Ainv = t(f)%*%f/sigma^2 + prior_inv
    A = chol2inv(chol(Ainv))
    b = t(f)%*%y/sigma^2 + prior_inv%*%beta_matrix[j,]
    Ab = A%*%b
    
    # Get predictions
    predmean[h] = f_matrix[h,]%*%Ab
    predstd[h] = sqrt(diag(f_matrix[h,]%*%A%*%t(f_matrix[h,])))
  }
  outpred = list(predmean = predmean, predstd = predstd)
  return(outpred)
}

# Soft post predict
soft_post_predict = function(f_matrix, beta_matrix, tau_matrix, phi_matrix, y_vec, sigma){
  # Define terms
  K = ncol(f_matrix)
  n = nrow(f_matrix)
  B = nrow(beta_matrix)
  y = y_vec
  predmean = predstd = rep(0,n)
  f = matrix(0,nrow=n,ncol=0)
  tau_vec = NULL
  beta_vec = NULL
  for(j in 1:B){
    # Terms
    f = cbind(f,phi_matrix[j,]*f_matrix)
    tau_vec = c(tau_vec,tau_matrix[j,])
    beta_vec = c(beta_vec,beta_matrix[j,])
  }
 
  # Mean and Variance
  prior_inv = diag(1/tau_vec^2)
  Ainv = t(f)%*%f/sigma^2 + prior_inv
  A = chol2inv(chol(Ainv))
  b = t(f)%*%y/sigma^2 + prior_inv%*%beta_vec
  Ab = A%*%b
  
  # Get predictions
  predmean = f%*%Ab
  predstd = sqrt(diag(f%*%A%*%t(f)))

  outpred = list(predmean = predmean, predstd = predstd)
  return(outpred)
}

# Soft posterior mean and std
soft_post = function(f_matrix, beta_matrix, tau_matrix, phi_matrix, y_vec, sigma){
  # Define terms
  K = ncol(f_matrix)
  n = nrow(f_matrix)
  B = nrow(beta_matrix)
  y = y_vec
  predmean = predstd = rep(0,n)
  f = matrix(0,nrow=n,ncol=0)
  tau_vec = NULL
  beta_vec = NULL
  for(j in 1:B){
    # Terms
    f = cbind(f,phi_matrix[j,]*f_matrix)
    tau_vec = c(tau_vec,tau_matrix[j,])
    beta_vec = c(beta_vec,beta_matrix[j,])
  }
  
  # Mean and Variance
  prior_inv = diag(1/tau_vec^2)
  Ainv = t(f)%*%f/sigma^2 + prior_inv
  A = chol2inv(chol(Ainv))
  b = t(f)%*%y/sigma^2 + prior_inv%*%beta_vec
  Ab = A%*%b
  
  outpost = list(postmean = Ab, poststd = A)
  return(outpost)
}

# Regular BMM Post
mu_post = function(f_matrix, tnode_list, beta_matrix, tau_matrix, y_vec, sigma){
  # Define terms
  K = ncol(f_matrix)
  n = nrow(f_matrix)
  B = length(tnode_list)
  mumean = mucov = vector('list',B)
  # Loop through tnodes
  for(j in 1:B){
    # Terms
    h = tnode_list[[j]]
    f = f_matrix[h,]
    y = y_vec[h] 
    prior_inv = diag(1/tau_matrix[j,]^2)
    
    # Mean and Variance
    Ainv = t(f)%*%f/sigma^2 + prior_inv
    A = chol2inv(chol(Ainv))
    b = t(f)%*%y/sigma^2 + prior_inv%*%beta_matrix[j,]
    Ab = A%*%b
    
    # Get mu post
    mumean[[j]] = Ab
    mucov[[j]] = A
  }
  outpred = list(mu_mean = mumean, mu_cov = mucov)
  return(outpred)
}

# Soft Z post
soft_z_pred = function(f_matrix, phix, mu_mean_list){
  # Setup
  B = ncol(phix)
  K = ncol(f_matrix)
  n = nrow(f_matrix)
  wts_matrix = matrix(0,nrow = n, ncol= K)
  # Get wts
  for(j in 1:B){
    wts_matrix = wts_matrix +  phix[,j]*matrix(mu_mean_list[[j]], nrow = n, ncol = K, byrow = TRUE)  
  }
  # Get predction
  outpred = list(pred_mean = rowSums(f_matrix*wts_matrix), wts_mean = wts_matrix)
  return(outpred)
}

#-----------------------------------------------------
# Terminal node MLE functions
#-----------------------------------------------------
mle_predict = function(f_matrix, tnode_list, y_vec, sigma){
  # Setup
  n = nrow(f_matrix)
  B = length(tnode_list)
  predmean = predstd = rep(0,n)
  mle_mu = list()
  mle_cov = list()
  # Get predictions by terminal node
  for(b in 1:B){
    h = tnode_list[[b]]
    f = f_matrix[h,]
    y = y_vec[h]
    ftf_inv = solve(t(f)%*%f)
    mu_hat = ftf_inv%*%t(f)%*%y 
    predmean[h] = f%*%mu_hat
    predstd[h] = sqrt(diag(f%*%(ftf_inv)%*%t(f)))*sigma
    mle_mu[[b]] = mu_hat
    mle_cov[[b]] = ftf_inv/sigma^2
  }
  outpred = list(predmean = predmean, predstd = predstd, mle_mu = mle_mu, mle_cov = mle_cov)
  return(outpred)
}

#-----------------------------------------------------
# Terminal Node Plot Predictions
#-----------------------------------------------------
plot_predict = function(x,y,predmean, predstd, tnode_list=NULL, ci = 0.95, y_lim=NULL, title = "Predict", color = 'black', bounds = TRUE){
  # Setup and get bounds
  alpha = (1-ci)/2
  z = qnorm(1-alpha,0,1)
  ub = predmean + z*predstd
  lb = predmean - z*predstd
  
  # Set up range
  rng = max(predmean) - min(predmean)
  if(is.null(y_lim)){y_lim = c(min(predmean)-rng*0.25, max(predmean))+rng*0.25}
  
  # Plot
  if(is.null(tnode_list)){
    plot(x, predmean, xlab = 'X', ylab = 'Y', main = title, panel.first = {grid(col = 'lightgrey')}, type = 'l',lwd = 2, col = color, ylim = y_lim)
    points(x,y, cex = 0.8, pch = 16)
    if(bounds){
      lines(x, lb, col = color, lty = 'dashed')
      lines(x, ub, col = color, lty = 'dashed')
    }
  }else{
    # Plot
    plot(x, y, xlab = 'X', ylab = 'Y', main = title, panel.first = {grid(col = 'lightgrey')}, pch = 16, ylim = y_lim)
    B = length(tnode_list)
    for(b in 1:B){
      h = tnode_list[[b]]
      lines(x[h],predmean[h], lwd = 2, col = color)
      if(bounds){
        lines(x[h], lb[h], col = color, lty = 'dashed')
        lines(x[h], ub[h], col = color, lty = 'dashed')
      }
    }
  }

}


#-----------------------------------------------------
# Posterior Distributions
#-----------------------------------------------------
# Log unnormalized posterior of beta
post_beta_logun = function(yvec, f_matrix, sigma, tau, m){
  # Define terms
  s2 = sigma^2
  t2 = tau^2
  f = f_matrix
  K = ncol(f)
  prior_inv = diag(1/t2,K)
  yvec = matrix(yvec,ncol=1)
  
  # Get cov matrix
  Ainv = t(f)%*%f/sigma^2 + prior_inv
  A = chol2inv(chol(Ainv))
  
  beta_set = diag(1/m,K)
  fys = t(f)%*%yvec/s2
  out = 0
  for(i in 1:K){
    b = beta_set[,i]
    out[i] = -0.5/t2*(t(b)%*%b - t(b)%*%A%*%b/t2 - 2*t(b)%*%A%*%fys)
    #out[i] = 0.5*t(beta_set[,i]/t2 + fys)%*%A%*%(beta_set[,i]/t2 + fys)
  }
  return(out)
}


#-----------------------------------------------------
# Log Marginal likelihood and marginal priors -- computes densities at input y (or mu)
#-----------------------------------------------------
# Marginal Selection prior -- GMM after integrating over beta and mu
lm_select = function(muvec,num_models,k=2, m=1,beta_prior=NULL){
  # Set beta_prior
  if(is.null(beta_prior)){beta_prior = rep(1/num_models, num_models)}
  if(length(beta_prior!=num_models)){stop("Incorrect dimension for beta_prior. Must be of length num_models!")}
  
  # Get GMM
  den = 0
  for(j in 1:num_models){
    out = select_prior(num_models = num_models,sc=j,k=k,m=m)
    den = den + beta_prior[j]*prod(dnorm(muvec,out$beta,out$tau)) #components of mu are indep apriori 
  }
  
  return(log(den))
}


#-----------------------------------------------------
# Plotting Functions
#-----------------------------------------------------

