#example_model_mixing.R
setwd("/home/johnyannotty/Documents/Open BT Project SRC")

# Load the R wrapper functions to the OpenBT library.
source("/home/johnyannotty/Documents/Open BT Project SRC/openbt.R")
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

#Construct the small-g discrepancy
dsg = function(g,ns){
  #Get the term number
  k = 0:ns
  
  #Get the coefficients -- only even coefficients are non-zero
  sk = ifelse(k %% 2 == 0,sqrt(2)*gamma(k + 1/2)*(-4)^(k/2)/factorial(k/2), 0)
  
  #Estimate cbar
  cbar = sqrt(mean(sk^2))
  
  #get the variance estimate 
  if((ns/2)%%2 == 0){
    v = (cbar^2)*(factorial(ns/2 + 2)^2)*g^(ns + 4)
  }else{
    v = (cbar^2)*(factorial(ns/2 + 1)^2)*g^(ns + 2)
  }
  s = sqrt(v)
  return(s)
}

#Construct the large-g discrepancy
dlg = function(g,nl){
  #Get the term number
  k = 0:nl
  
  #Get the coefficients
  lk = gamma(k/2 + 0.25)*(-0.5)^k/(2*factorial(k))
  
  #Estimate cbar
  if(nl < 2){
    print("Warning, nl < 2, dbar is not estimated as intended")
    dbar = 1
  }else{
    #Estimate cbar using coefs of order 2 through nl
    dbar = sqrt(mean(lk[-c(1,2)]^2))  
  }
  
  #Get standard deviation
  v = (dbar^2)*(1/(factorial(nl+1)^2))*(1/g^(2*nl + 3))  
  s = sqrt(v)
  return(s)
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
       ylim = c(1,4), type = 'l', panel.first = {grid(col = 'lightgrey')})
  lines(g_grid, fs, col = 'red', lty = 'dashed')
  lines(g_grid, fl, col = 'blue', lty = 'dashed')
  legend('bottomright', legend = c('F(g)', fs_name, fl_name), cex = 0.75, 
         lty = c(1, 2, 2), col = c('black', 'red', 'blue'))
}

plot_exp(g_grid, ns = 2, nl = 2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Generate training data
#Set parameters
n_train = 20
n_test = 200
s = 0.03 #s = 0.03

set.seed(4321)
x_train = seq(0.05, 0.5, length = n_train)
y_train = fg(x_train) + rnorm(n_train, 0, s)

#Set a grid of test points
x_test = seq(0.05, 0.5, length = n_test)
fg_test = fg(x_test)
fs_test = sapply(x_test, fsg, 2)
fl_test = sapply(x_test, flg, 4)

#Plot the training data
plot(x_train, y_train, pch = 16, cex = 0.8, main = 'Training data')
lines(g_grid, fg(g_grid))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Define the Model set -- a matrix where each column is an the output evaluated at the train/test pts
#small_g = c(2,4,6)
#large_g = c(3,6,9)
small_g = 2 #Which small-g expansions to use
large_g = 4 #Which large-g expansions to use
g_exp = c(small_g, large_g) #Mix both small and large g
K = length(g_exp) 
Ks = length(small_g) #Number of small-g models
Kl = length(large_g) #Number of large-g models

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
  if(i <= length(small_g)){
    f_train[,i] = sapply(x_train, fsg, ns = g_exp[i])
    f_test[,i] = sapply(x_test, fsg, ns = g_exp[i])
    
    f_train_dsd[,i] = sapply(x_train, dsg, ns = g_exp[i])
    f_test_dsd[,i] = sapply(x_test, dsg, ns = g_exp[i])
  }else{
    #Get the large g expansion output  
    f_train[,i] = sapply(x_train, flg, nl = g_exp[i])
    f_test[,i] = sapply(x_test, flg, nl = g_exp[i])
    
    f_train_dsd[,i] = sapply(x_train, dlg, nl = g_exp[i])
    f_test_dsd[,i] = sapply(x_test, dlg, nl = g_exp[i])
  }
}

#Cast x_train and x_test as matrices
x_train = as.matrix(x_train, ncol = 1)
x_test = as.matrix(x_test, ncol = 1)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Model Mixing with OpenBT 
fit=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),ntree = 10,ntreeh=1,numcut=100,tc=4,model="mixbart",modelname="physics_model",
           ndpost = 10000, nskip = 1000, nadapt = 5000, adaptevery = 500, printevery = 500,
           power = 1.0, minnumbot = 5, overallsd = sd(y_train)/sqrt(4), k = 2)

#Get mixed mean function
fitp=predict.openbt(fit,x.test = x_test, f.test = f_test,tc=4)

plot(g_grid, fg(g_grid), pch = 16, cex = 0.8, main = 'Fits', type = 'l', ylim = c(1.8,2.8))
points(x_train, y_train, pch = 3)
lines(g_grid, sapply(g_grid,fsg, small_g), col = 'red', lty = 2)
lines(g_grid, sapply(g_grid,flg, large_g), col = 'blue', lty = 2)
points(x_test, fitp$mmean, col = 'green4')
points(x_test, fitp$m.lower, col = 'orange')
points(x_test, fitp$m.upper, col = 'orange')

#Get the model weights
fitw = openbt.mixingwts(fit, x.test = x_test, numwts = 2, tc = 4)

#Plot model weights
plot(x_test, fitw$wmean[,1], pch = 16, col = 'red', type = 'o', ylim = c(-1,1))
points(x_test, fitw$wmean[,2], col = 'blue', pch = 16)
lines(x_test, fitw$w.upper[,1], col = 'red', lty = 'dashed')
lines(x_test, fitw$w.lower[,1], col = 'red', lty = 'dashed')
lines(x_test, fitw$w.upper[,2], col = 'blue', lty = 'dashed')
lines(x_test, fitw$w.lower[,2], col = 'blue', lty = 'dashed')

#Trace Plot of the weights
plot(fitw$wdraws[[1]][,25], type = 'l')
plot(fitw$wdraws[[1]][,185], type = 'l')
plot(fitw$wdraws[[2]][,25], type = 'l')
plot(fitw$wdraws[[2]][,185], type = 'l')

#Trace plots of thetas
tnp10 = read.table(paste0(fit$folder,"/physics_model.tnp1draws0"))
tnp13 = read.table(paste0(fit$folder,"/physics_model.tnp1draws3"))
tnp20 = read.table(paste0(fit$folder,"/physics_model.tnp2draws0"))
tnp23 = read.table(paste0(fit$folder,"/physics_model.tnp2draws3"))
tnp_list = list(tnp10, tnp13, tnp20, tnp23)

par(mfrow = c(2,3))
for(i in 1:4){
  for(j in 1:5){
    temp = tnp_list[[i]] 
    plot(temp[,j], type ='l')
  }  
}

#Save results
fitp_out = list(mmean = fitp$mmean,m.lower = fitp$m.lower, m.upper = fitp$m.upper)
fitw_out = list(wmean = fitw$wmean,w.lower = fitw$w.lower, w.upper = fitw$w.upper, wdraws = list(fitw$wdraws[[1]][,c(10,100,190)],fitw$wdraws[[2]][,c(10,100,190)] ))
fittpn_out =  tnp_list
#saveRDS(fitp_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Mixing Wts Variability 02-17/sg2lg4_p.rds")
#saveRDS(fitw_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Mixing Wts Variability 02-17/sg2lg4_w.rds")
#saveRDS(fittpn_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Mixing Wts Variability 02-17/sg2lg4_tnp.rds")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Model Mixing with OpenBT - With Discrepancy
fit=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),ntree = 30,ntreeh=1,numcut=100,tc=4,model="mixbart",modelname="physics_model",
           ndpost = 10000, nskip = 1000, nadapt = 5000, adaptevery = 500, printevery = 500,
           power = 1.0, minnumbot = 5, overallsd = sd(y_train)/sqrt(4), k = 1.75,
           f.discrep.mean =  f_train_dmean, f.discrep.sd = f_train_dsd)

#Get mixed mean function
fitp=predict.openbt(fit,x.test = x_test, f.test = f_test, f.discrep.mean =  f_test_dmean, f.discrep.sd = f_test_dsd ,tc=4)

plot(g_grid, fg(g_grid), pch = 16, cex = 0.8, main = 'Fits', type = 'l', ylim = c(1.0,5.8))
points(x_train, y_train, pch = 3)
lines(g_grid, sapply(g_grid,fsg, small_g), col = 'red', lty = 2)
lines(g_grid, sapply(g_grid,flg, large_g), col = 'blue', lty = 2)
points(x_test, fitp$mmean, col = 'green4')
points(x_test, fitp$m.lower, col = 'orange')
points(x_test, fitp$m.upper, col = 'orange')
for(i in 1:n_test){
  if(i%%5 == 0 && i%%10 != 0){
    lb1 = f_test[i,1] - 2*f_test_dsd[i,1]
    ub1 = f_test[i,1] + 2*f_test_dsd[i,1] 
    lines(rep(x_test[i,],2), c(lb1, ub1), col = 'red', lty = 2)
  }
  
  if(i%%10 == 0){
    lb2 = f_test[i,2] - 2*f_test_dsd[i,2]
    ub2 = f_test[i,2] + 2*f_test_dsd[i,2]
    lines(rep(x_test[i,],2), c(lb2, ub2), col = 'blue', lty = 2)
  }
  
}

#Get the model weights
fitw = openbt.mixingwts(fit, x.test = x_test, numwts = 2)

#Plot model weights
plot(x_test, fitw$wmean[,1], pch = 16, col = 'red', type = 'o', ylim = c(-1,2))
points(x_test, fitw$wmean[,2], col = 'blue', pch = 16)
lines(x_test, fitw$w.upper[,1], col = 'red', lty = 'dashed')
lines(x_test, fitw$w.lower[,1], col = 'red', lty = 'dashed')
lines(x_test, fitw$w.upper[,2], col = 'blue', lty = 'dashed')
lines(x_test, fitw$w.lower[,2], col = 'blue', lty = 'dashed')

#Save the results 
#openbt.save(fit,"/home/johnyannotty/Documents/Model Mixing BART/smallg2largeg4")
fitp_out = list(mmean = fitp$mmean,m.lower = fitp$m.lower, m.upper = fitp$m.upper)
fitw_out = list(wmean = fitw$wmean,w.lower = fitw$w.lower, w.upper = fitw$w.upper)
#saveRDS(fitp_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Results 02-08/sg4lg4_2p.rds")
#saveRDS(fitw_out, "/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/Results 02-08/sg4lg4_2w.rds")

#physics_data = data.frame(x_train = x_train, y_train = y_train)
#write.csv(physics_data, '/home/johnyannotty/Documents/Model Mixing BART/Open BT Examples/scale and shift 01-25/sg4lg4_train.csv',row.names = FALSE)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Check calculations
inv_prior_var = diag(c(50,50))
beta0 = c(0,0)
data_rng = c(11:20)
sig2 = 0.1
solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)

sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[1,1])
sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[2,2])

b = solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
f_train[data_rng,]%*%b

bhat = solve(t(f_train[data_rng,])%*%f_train[data_rng,])%*%t(f_train[data_rng,])%*%y_train[data_rng]
f_train[data_rng,]%*%bhat

#G prior
g = 10
q = g/(g+1)
bhat = solve(t(f_train[data_rng,])%*%f_train[data_rng,])%*%t(f_train[data_rng,])%*%y_train[data_rng]
q*bhat

#use discrepancies for the prior
data_rng = c(11:20)
inv_prior_var = 2*diag(c(sum(f_train_dsd[data_rng,1]),sum(f_train_dsd[data_rng,2])))
beta0 = c(0,0)
sig2 = 1
#solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
#solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)

#sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[1,1])
#sqrt(solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)[2,2])

b = solve(t(f_train[data_rng,])%*%f_train[data_rng,]/sig2 + inv_prior_var)%*%(t(f_train[data_rng,])%*%y_train[data_rng]/sig2 + inv_prior_var%*%beta0)
f_train[data_rng,]%*%b
bhat = solve(t(f_train[data_rng,])%*%f_train[data_rng,])%*%t(f_train[data_rng,])%*%y_train[data_rng]
f_train[data_rng,]%*%bhat
