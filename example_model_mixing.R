#example_model_mixing.R
setwd("/home/johnyannotty/Documents/Open BT Project SRC")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Construct the required functions
#True function
fg = function(g){
  x = 1/(32*g^2)
  K = besselK(x = x, nu = 0.25)
  b = exp(x)/(2*sqrt(2)*g)*K
  return(b)
}

#Construct the small-g expansion
fsg = function(g, ns){
  #Get the term number
  k = 0:ns
  
  #Get the coefficients -- only even coefficients are non-zero
  sk = ifelse(k %% 2 == 0,sqrt(2)*gamma(k + 1/2)*(-4)^(k/2)/factorial(k/2), 0)
  
  #Get the expansion
  f = sum(sk*g^k)
  return(f)
}

#Construct the large-g expansion
flg = function(g, nl){
  #Get the term number
  k = 0:nl
  
  #Get the coefficients
  lk = gamma(k/2 + 0.25)*(-0.5)^k/(2*factorial(k))
  
  #Get the expansion
  f = sum(lk*g^(-k))/sqrt(g)
  return(f)
}

#Test out the functions and approximations
fg(0.1)
fsg(0.1, 4)
fg(0.3)
flg(0.3, 5)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Graph the expansions and the true function.
#Create a grid of g's
g_grid = seq(0.01, 0.5, length = 300)

#Get a simple plot using only one large and one small g expansion
plot_exp = function(g_grid, ns, nl){
  #Get labels
  fs_name = paste0('Fs(',ns,')')
  fl_name = paste0('Fl(',nl,')')
  #Evaluate the functions 
  f = fg(g_grid)
  fs = sapply(g_grid, fsg, ns = ns)
  fl = sapply(g_grid, flg, nl = nl)
  plot(g_grid, f, main = 'F(g) and Exansions vs. g', xlab = 'g', ylab = 'F(g)', 
       ylim = c(0,4), type = 'l', panel.first = {grid(col = 'lightgrey')})
  lines(g_grid, fs, col = 'red', lty = 'dashed')
  lines(g_grid, fl, col = 'blue', lty = 'dashed')
  legend('bottomright', legend = c('F(g)', fs_name, fl_name), cex = 0.75, 
         lty = c(1, 2, 2), col = c('black', 'red', 'blue'))
}

plot_exp(g_grid, ns = 2, nl = 4)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Generate training data
#Set parameters
n_train = 40 
n_test = 300
s = 0.03 #s = 0.03

set.seed(44332211)
x_train = seq(0.01, 0.5, length = n_train)
y_train = fg(x_train) + rnorm(n_train, 0, s)

#Set a grid of test points
x_test = seq(0.01, 0.5, length = n_test)
fg_test = fg(x_test)

#Plot the training data
plot(x_train, y_train, pch = 16, cex = 0.8, main = 'Training data')
lines(g_grid, fg(g_grid))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Define the Model set -- a matrix where each column is an the output evaluated at the train/test pts
#small_g = c(2,4,6)
#large_g = c(3,6,9)
small_g = 2 #Which small-g expansions to use
large_g = 6 #Which large-g expansions to use
g_exp = c(small_g, large_g) #Mix both small and large g
K = length(g_exp) 
Ks = length(small_g) #Number of small-g models
Kl = length(large_g) #Number of large-g models

#Define matrices to store the function values
f_train = matrix(0, nrow = n_train, ncol = K)
f_test = matrix(0, nrow = n_test, ncol = K)

#Computation
for(i in 1:K){
  #Get the small g expansion output
  if(i <= length(small_g)){
    f_train[,i] = sapply(x_train, fsg, ns = g_exp[i])
    f_test[,i] = sapply(x_test, fsg, ns = g_exp[i])
  }else{
    #Get the large g expansion output  
    f_train[,i] = sapply(x_train, flg, nl = g_exp[i])
    f_test[,i] = sapply(x_test, flg, nl = g_exp[i])
  }
}

#Cast x_train and x_test as matrices
x_train = as.matrix(x_train, ncol = 1)
x_test = as.matrix(x_test, ncol = 1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Model Mixing with OpenBT
# Load the R wrapper functions to the OpenBT library.
source("/home/johnyannotty/Documents/Open BT Project SRC/openbt.R")

# Model Mixing BART model
fit=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),ntree = 10,ntreeh=1,numcut=100,tc=4,model="mixbart",modelname="physics_model")

