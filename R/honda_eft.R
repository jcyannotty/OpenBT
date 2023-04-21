#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: John Yannotty (yannotty.1@buckeyemail.osu.edu)
# Desc: R implementation of Honda's Prototype EFTs. Motivated after Alexandra Semposki's
#       SAMBA package written in python, which also implements these models. This R script 
#       implements the non-informative error model model for the truncation errors. 
# Version: 2.0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Construct the required functions
# Get the true function
fg = function(g){
  x = 1/(32*g^2)
  K = besselK(x = x, nu = 0.25)
  b = exp(x)/(2*sqrt(2)*g)*K
  return(b)
}

# Construct the small-g expansion
fsg = function(g, ns){
  #Get the term number
  k = 0:ns
  
  #Get the coefficients -- only even coefficients are non-zero
  sk = ifelse(k %% 2 == 0,sqrt(2)*gamma(k + 1/2)*(-4)^(k/2)/factorial(k/2), 0)
  
  #Get the expansion
  f = sum(sk*g^k)
  return(f)
}

# Construct the large-g expansion
flg = function(g, nl){
  #Get the term number
  k = 0:nl
  
  #Get the coefficients
  lk = gamma(k/2 + 0.25)*(-0.5)^k/(2*factorial(k))
  
  #Get the expansion
  f = sum(lk*g^(-k))/sqrt(g)
  return(f)
}

# Compute cbar for the sg truncation error
get_cbar = function(g, ns){
  #Get the term number
  k = 0:ns
  #Get the coefficients -- only even coefficients are non-zero
  sk = ifelse(k %% 2 == 0,sqrt(2)*gamma(k + 1/2)*(-4)^(k/2)/(factorial(k/2)*factorial(k)), 0)
  #Estimate cbar
  h = which(sk!=0)
  cbar = sqrt(mean((sk[h]^2)))
  return(cbar)
}

# Construct the small-g discrepancy - returns the std deviation
# assumes a mean of zero for the errors 
dsg = function(g,ns,cbar = NULL){
  if(is.null(cbar)){
    #cbar = sqrt(mean(sk[1:(ns/2)]^2))
    cbar = get_cbar(g, ns)
  }
  
  if((ns)%%2 == 0){
    v = (cbar^2)*(factorial(ns + 2)^2)*g^(2*ns + 4)
  }else{
    v = (cbar^2)*(factorial(ns + 1)^2)*g^(2*ns + 2)
  }
  s = sqrt(v)
  return(s)
}

# Compute dbar for the lg truncation error
get_dbar = function(g, nl){
  #Get the term number
  k = 0:(nl)
  
  #Get the coefficients
  lk = gamma(k/2 + 0.25)*(-0.5)^k*factorial(k)/(2*factorial(k))
  
  #Estimate dbar
  if(nl < 2){
    print("Warning, nl < 2, dbar is not estimated as intended")
    dbar = 1
  }else{
    #Estimate cbar using coefs of order 2 through nl
    dbar = sqrt(mean(lk[-c(1,2)]^2))  
  }
  return(dbar)
}

# Construct the large-g discrepancy - returns the std deviation
# assumes a mean of zero for the errors
dlg = function(g,nl,dbar = NULL){
  if(is.null(dbar)){
    #Estimate dbar
    if(nl < 2){
      print("Warning, nl < 2, dbar is not estimated as intended")
      dbar = 1
    }else{
      #Estimate cbar using coefs of order 2 through nl
      #dbar = sqrt(mean(lk[-c(1,2)]^2))
      dbar = get_dbar(g,nl)
    }
  }
  #Get standard deviation
  v = (dbar^2)*(1/(factorial(nl+1)^2))*(1/g^(2*nl + 3))  
  s = sqrt(v)
  return(s)
}

# Generate training and testing data
# n_train = training size
# n_test = testing size
# s = error standard deviation
# sg = int or list of sg orders, can also pass NULL for no sg models
# lg = int or list of lg orders, can also pass NULL for no lg models
# minx, maxx = min/max values for the domain of x
# seed = value, used to reproduce results
# random_x = (bool) do you want to generate randomly spaced x's?
# x_eft_train = locations of the model runs to use to "fit" the EFT -- used to learn cbar or dbar
get_data = function(n_train, n_test, s, sg, lg, minx, maxx, seed, random_x = FALSE, x_eft_train = NULL, x_train = NULL){
  # Find cbar and dbar if x_eft_train = NULL
  if(is.null(x_eft_train)){
    cbar = NULL
    dbar = NULL
  }else{
    cbar = dbar =  1
    if(!is.null(sg)){
      for(i in 1:length(sg)){
        cbar[i] = get_cbar(x_eft_train, sg[i]) 
      }  
    }else{
      cbar = NULL
    }
    if(!is.null(lg)){
      for(i in 1:length(lg)){
        dbar[i] = get_dbar(x_eft_train, lg[i]) 
      }  
    }else{
      dbar = NULL
    }
  }
  
  #Get test and train data
  set.seed(seed)
  if(random_x){
    x_train = runif(n_train, minx, maxx)
    x_train = x_train[order(x_train)]
  }else if(is.null(x_train)){
    x_train = seq(minx, maxx, length = n_train)
  }
  fg_train = fg(x_train) 
  y_train = fg(x_train) + rnorm(n_train, 0, s)
  
  #Set a grid of test points
  x_test = seq(minx, maxx, length = n_test)
  fg_test = fg(x_test)
  y_test = fg_test + rnorm(n_test, 0, s)
  
  #Define the Model set -- a matrix where each column is an the output evaluated at the train/test pts
  g_exp = c(sg, lg) #Mix both small and large g
  K = length(g_exp) 
  Ks = length(sg) #Number of small-g models
  Kl = length(lg) #Number of large-g models
  
  #Define matrices to store the function values
  f_train = matrix(0, nrow = n_train, ncol = K)
  f_test = matrix(0, nrow = n_test, ncol = K)
  
  #Define discrepancy information
  f_train_dmean = matrix(0, nrow = n_train, ncol = K)
  f_train_dsd = matrix(1, nrow = n_train, ncol = K)
  f_test_dmean = matrix(0, nrow = n_test, ncol = K)
  f_test_dsd = matrix(1, nrow = n_test, ncol = K)
  
  #Computation
  for(i in 1:K){
    #Get the small g expansion output
    if(i <= length(sg)){
      f_train[,i] = sapply(x_train, fsg, ns = g_exp[i])
      f_test[,i] = sapply(x_test, fsg, ns = g_exp[i])
      
      if(is.null(cbar)){
        cbar0 = get_cbar(x_train, g_exp[i])
      }else{
        cbar0 = cbar[i]
      }
      f_train_dsd[,i] = sapply(x_train, dsg, ns = g_exp[i], cbar0)
      f_test_dsd[,i] = sapply(x_test, dsg, ns = g_exp[i], cbar0)
      
    }else{
      #Get the large g expansion output  
      f_train[,i] = sapply(x_train, flg, nl = g_exp[i])
      f_test[,i] = sapply(x_test, flg, nl = g_exp[i])
      
      if(is.null(dbar)){
        dbar0 = get_dbar(x_train, g_exp[i])
      }else{
        dbar0 = dbar[i-length(sg)]
      }
      f_train_dsd[,i] = sapply(x_train, dlg, nl = g_exp[i],dbar0)
      f_test_dsd[,i] = sapply(x_test, dlg, nl = g_exp[i],dbar0)
    }
  }  
  
  out = list(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, f_train = f_train, 
             f_test = f_test,fg_train = fg_train, fg_test = fg_test, f_train_dmean = f_train_dmean, f_train_dsd = f_train_dsd,
             f_test_dmean = f_test_dmean, f_test_dsd = f_test_dsd)
  return(out)
}
