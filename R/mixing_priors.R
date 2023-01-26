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


#-----------------------------------------------------
# Terminal node MLE functions
#-----------------------------------------------------
mle_predict = function(f_matrix, tnode_list, y_vec, sigma){
  # Setup
  n = nrow(f_matrix)
  B = length(tnode_list)
  predmean = predstd = rep(0,n)
  # Get predictions by terminal node
  for(b in 1:B){
    h = tnode_list[[b]]
    f = f_matrix[h,]
    y = y_vec[h]
    ftf_inv = solve(t(f)%*%f)
    mu_hat = ftf_inv%*%t(f)%*%y 
    predmean[h] = f%*%mu_hat
    predstd[h] = sqrt(diag(f%*%(ftf_inv)%*%t(f)))*sigma
  }
  outpred = list(predmean = predmean, predstd = predstd)
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
# Terminal node draw functions
#-----------------------------------------------------


#-----------------------------------------------------
# Log Marginal likelihood and marginal priors -- computes densities at input y (or mu)
#-----------------------------------------------------
# Marginal Selection prior -- GMM after integrating over beta
lm_selet_mu = function(muvec,num_models,k=2, m=1,beta_prior=NULL){
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