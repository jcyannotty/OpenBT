#-----------------------------------------------------
# Polynomial model functions
# Author: John Yannotty (yannotty.1@buckeyemail.osu.edu)
# Version: 1.0
# Description: R module which defines polynomial functions which
#   are used in toy model mixing problems
#-----------------------------------------------------
#-----------------------------------------------------
# Polynomial functions of order k -- denoted by fp 
fp = function(xvec, a=0, b = 0, c = 1, p = 1){return(c*(xvec-a)^p + b)}

# Piecewise function requires matrix of inputs (nxK) where K-1 = number of knots 
fpw = function(xvec, knots, fmatrix = NULL){
  n = length(xvec)
  fout = rep(0,n)
  for(k in 1:length(knots)){
    if(k == 1){
      h = which(xvec <= knots[k])
      fout[h] = fmatrix[h,k]
    }
    if(k <= length(knots) & k > 1){
      h = which(xvec <= knots[k] & xvec > knots[k-1])
      fout[h] = fmatrix[h,k]
    }
    if(k == length(knots)){
      h = which(xvec > knots[k])
      fout[h] = fmatrix[h,k+1]
    }
  }
  return(fout)
}

# Softmax transformation -- basis = nxK matrix
softmax = function(basis){
  wts = exp(basis)/rowSums(exp(basis))
  colnames(wts) = paste0("w",1:ncol(basis))
  return(wts)
}

# Generate weight basis for a model M_k
# a,b,c, and p can be vectors which match ncol of x
get_wt_basis = function(x, a=0,b=0, c=1, p=1){
  # Initialize basis matrix
  if(is.matrix(x)){ 
    px = ncol(x)
    n = nrow(x)
  }else{
    px = 1
    n = length(x)
    x = matrix(x, ncol = 1, nrow = n)
  }
  wts_basis = rep(0,n)

  # Loop through x cols to generate basis
  if(length(a)<px){
    cat("Setting a = 0 for x dimensions ... ",(px-length(a)):px)
    a = c(a,rep(0,px-length(a)))
  }
  if(length(b)<px){
    cat("Setting b = 0 for x dimensions ... ",(px-length(b)):px)
    b = c(b,rep(0,px-length(b)))
  }
  if(length(a)<px){
    cat("Setting c = 1 for x dimensions ... ",(px-length(c)):px)
    c = c(c,rep(0,px-length(c)))
  }
  if(length(p)<px){
    cat("Setting p = 1 for x dimensions ... ",(px-length(p)):px)
    p = c(p,rep(0,px-length(p)))
  }
  
  for(j in 1:px){
    wts_basis = wts_basis + fp(x[,j],a = a[j],b = b[j],c = c[j], p=p[j])    
  }
  
  return(wts_basis)
}

# Get fmatrix
get_fmatrix = function(x, h_list, wts_matrix, bias_vec){
  f_matrix = wts_matrix # easy initialization
  for(i in 1:length(h_list)){
    f_matrix[,i] = fp(x, a=h_list[[i]]$a, b=h_list[[i]]$b, c=h_list[[i]]$c, p=h_list[[i]]$p)  
  }
  return(f_matrix)
}

# Generate true model using a mixture of simulators and linear bias 
get_ftrue = function(fmatrix, wmatrix, biasvec){
  f = rowSums(fmatrix*wmatrix) + biasvec
  return(f)
}
