#-----------------------------------------------------
# Orthogonal functions for calibrations
# Author: John Yannotty (yannotty.1@buckeyemail.osu.edu)
# Version: 1.0
# Description: R module which defines a set of orthogonal functions which
#   are used in toy model calibration problems
# Reference: https://demonstrations.wolfram.com/OrthogonalityOfTwoFunctionsWithWeightedInnerProducts/
#-----------------------------------------------------
#-----------------------------------------------------
# Sine and Cosine models over -pi to pi
sin_cos_orthmodel = function(xvec, eta = 's', delta = 'c', eta_scale = 1, delta_scale = 2){
  if(eta == tolower(eta)){
    eta_out = sin(xvec*eta_scale)
  }else{
    eta_out = cos(xvec*eta_scale)
  }
  
  if(delta == tolower(delta)){
    delta_out = cos(xvec*delta_scale)
  }else{
    delta_out = cos(xvec*delta_scale)
  }
  
  f_out = eta_out + delta_out
  out = list(eta = eta_out, delta = delta_out, f = f_out)
  return(f_out)
}



#-----------------------------------------------------
# Chebyshev Polynomials


#-----------------------------------------------------
# Legendre Polynomials