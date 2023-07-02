#------------------------------------------------
# Openbt calibration priors
#------------------------------------------------
# Orthogonal Calibration Prior
orthog_prior = function(Avec,scale,k2,m){
  tau2 = scale/(k2*sqrt(m))
  D = length(Avec)
  I = diag(1,D)
  tau2_matrix = tau2*(I - Avec%*%t(Avec) /sum(Avec*Avec))
  return(tau2_matrix)
}

# Compute sigma_c
get_sigmac = function(yc_train, m, k1){
  rgyc = range(yc_train)
  yc_mean = mean(yc_train)
  tau1 = (rgyc[2] - rgyc[1])/(2*k1*sqrt(m)) 
  avg_sse = mean((yc_train - yc_mean)^2)
  sigc2_mle = avg_sse - m*tau1^2 
}