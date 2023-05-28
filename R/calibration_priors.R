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
