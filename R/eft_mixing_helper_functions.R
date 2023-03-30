#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BART Model Mixing Helper Functions 
# Author: John Yannotty (yannotty.1@buckeyemail.osu.edu)
# Desc:
# Version: 1.0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(ggplot2)
library(dplyr)
library(grid)
library(gridExtra)
library(reshape2)
library(DescTools)

color_list = list(exp_cols = c("red","blue","green4"), mean_cols = c("purple","darkorchid2"), true_cols = "black",
                  wts_cols = c("red","blue","green4"))
gray_scale_list = list(exp_cols = c("gray60","gray60","gray60"), mean_cols = c("gray30","gray50"), true_cols = "gray60",
                       wts_cols = c("gray30","gray50","gray70") )

lty_list = list(exp_lty = c("dashed","dashed","dashed"), mean_lty = "solid", true_lty = "solid", wts_lty = c('solid', 'dashed', 'dotted'))
gs_lty_list = list(exp_lty = c("dashed","dotted","dotdash"), mean_lty = "solid", true_lty = "solid", wts_lty = c('solid', 'dashed', 'dotted'))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting -- GGplot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the predictions
plot_fit_gg2 = function(data,fitp, title = 'Non-Informative',y_lim = c(1,3),colors = color_list, 
                        line_type_list = lty_list, in_labs = NULL, grid_on = FALSE){
  df = data.frame(x_test = data$x_test, fg_test = data$fg_test,
                  data.frame(mmean = fitp$mmean,m.lower=fitp$m.lower,m.upper=fitp$m.upper),
                  data$f_test)
  n = nrow(df)
  
  K = ncol(data$f_test)
  exp_cols = (ncol(df) - K+1):ncol(df)
  
  colnames(df)[exp_cols] = paste0('f',1:K)
  exp_df_melt = melt(df, id.vars = "x_test", measure.vars = colnames(df)[exp_cols])
  df_melt = melt(df, id.vars = "x_test", measure.vars = c(colnames(df)[exp_cols],'fg_test','mmean'))
  #df_melt$line_group = c(rep('dashed',K*n),rep('solid', 2*n))
  #df_melt$size_group = c(rep('A',(K+1)*n),rep('B', n))
  
  line_group = c(line_type_list$exp_lty[1:K],line_type_list$mean_lty,line_type_list$true_lty)
  size_group = c(rep(1.1,(K+1)),1.2)
  
  if(is.null(in_labs)){in_labs = c(colnames(df)[exp_cols], "mean", 'true')}
  
  p = df_melt %>% ggplot() +
    geom_line(aes(x_test, value, color = variable, linetype = variable, size = variable)) + 
    scale_color_manual(values = c(colors$exp_cols[1:K], colors$true_cols, colors$mean_cols[1]), 
                       name = "", labels = in_labs) + 
    scale_linetype_manual(values = line_group, name = "", labels = in_labs) +
    scale_size_manual(values = size_group) +
    geom_ribbon(data = df, aes(x = x_test, 
                               ymin=m.lower,
                               ymax=m.upper),
                fill=colors$mean_cols[2], alpha=0.2) +
    theme_bw() +
    theme(axis.line = element_line(color = "grey70"),
          panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
          legend.position = "bottom", legend.text = element_text(size = 10)) +
    labs(x = "X", y = "F(x)", title = title) +
    geom_point(data = data.frame(data$x_train,data$y_train),aes(data$x_train,data$y_train), size= 1.0) + 
    coord_cartesian(ylim = y_lim) +
    guides(size = 'none')
  
  if(!grid_on){
    p = p + theme(panel.grid = element_blank())
  }
  return(p)
}


# Plot the weights
plot_wts_gg2 = function(fitw,x_test, title = 'Non-Informative',y_lim = c(0,1),colors = color_list, 
                        line_type_list = lty_list, gray_scale = FALSE, w_labs = NULL){
  K = ncol(fitw$wmean)
  colnames(fitw$wmean) = paste0("w",1:K)
  colnames(fitw$w.lower) = paste0("w",1:K)
  colnames(fitw$w.upper) = paste0("w",1:K)
  fitw$wmean = data.frame(x_test,fitw$wmean)
  wt_mean = reshape2::melt(fitw$wmean, id.vars = 'x_test', measure.vars = paste0("w",1:K))
  wt_lb = reshape2::melt(fitw$w.lower)
  wt_ub = reshape2::melt(fitw$w.upper)
  
  # Select last column in ub and lb
  wt_lb = wt_lb[,ncol(wt_lb)]
  wt_ub = wt_ub[,ncol(wt_ub)]
  
  wts_df = cbind(wt_mean, wt_lb, wt_ub)
  colnames(wts_df)[1:3] = c('x_test','wt','wt_mean')
  
  if(is.null(w_labs)){
    w_labs = c()
    for(j in 1:K){
      w_labs = c(w_labs,bquote(w[.(j)] * "(x)"))  
    }    
  }
  
  if(gray_scale){
    p = wts_df %>% ggplot() +
      geom_line(aes(x_test, wt_mean, color = wt, linetype = wt), size = 1.1) +
      geom_ribbon(aes(x = x_test, ymin=wt_lb, ymax=wt_ub, fill = wt), alpha=0.2) +
      theme_bw() +
      theme(axis.line = element_line(color = "grey70"),
            panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
            legend.position = 'bottom', legend.text = element_text(size = 12),
            panel.grid = element_blank()) +
      labs(x = "X", y = "W(x)", title = title,color = "Weights" ,fill = "Weights", linetype = 'Weights') +
      scale_color_manual(values = colors$exp_cols[1:K], labels = w_labs) +
      scale_linetype_manual(values = line_type_list$wts_lty[1:K], labels = w_labs) +
      scale_fill_manual(values = colors$exp_cols[1:K], labels = w_labs) + 
      coord_cartesian(ylim = y_lim) + 
      guides(fill = 'none') 
  }else{
    p = wts_df %>% ggplot() +
      geom_line(aes(x_test, wt_mean, color = wt), size = 1.1) +
      geom_ribbon(aes(x = x_test, ymin=wt_lb, ymax=wt_ub, fill = wt), alpha=0.2) +
      theme_bw() +
      theme(axis.line = element_line(color = "grey70"),
            panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
            legend.position = 'bottom', legend.text = element_text(size = 12)) +
      labs(x = "X", y = "W(x)", title = title,color = "Weights" ,fill = "Weights") +
      scale_color_manual(values = colors$exp_cols[1:K], labels = w_labs) + 
      scale_fill_manual(values = colors$exp_cols[1:K], labels = w_labs) + 
      coord_cartesian(ylim = y_lim)
  }
  return(p)
}

# Plot expansion
plot_exp_gg2 = function(ex_data,colors = color_list, line_type_list = lty_list, y_lim = c(0,5),in_labs = NULL, grid_on = FALSE){
  K = ncol(ex_data$f_test)
  df = data.frame(ex_data$x_test, ex_data$f_test, ex_data$fg_test)
  colnames(df) = c('x_test', paste0('f',1:K),'ftrue')
  if(is.null(in_labs)){in_labs = colnames(df)[-1]}
  df_melt = melt(df, id.vars = 'x_test')
  p = df_melt %>% ggplot() +
    geom_line(aes(x_test, value, color = variable, linetype = variable), size = 1.1) + 
    scale_color_manual(values = c(colors$exp_cols[1:K], colors$true_cols), labels = in_labs, name = "Models") +
    scale_linetype_manual(values = c(line_type_list$exp_lty[1:K], line_type_list$true_lty), labels = in_labs, name = "Models") +
    theme_bw() +
    theme(axis.line = element_line(color = "grey70"),
          panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
          legend.position = 'bottom', legend.text = element_text(size = 12)) +
    labs(x = "X", y = "F(x)", color = "Models") +
    coord_cartesian(ylim = y_lim)
  
  if(!grid_on){
    p = p + theme(panel.grid = element_blank())
  }
  return(p)
}

# Plot EFT with discrepancy
plot_eft_gg2 = function(x,eft_mean,eft_sd,color='red', ci=0.95, y_lim = c(0,5), title = "EFT Model"){
  # Get confidence interval
  alpha = (1 - ci)
  z = abs(qnorm(alpha/2,0,1))
  mean_lb = eft_mean - z*eft_sd
  mean_ub = eft_mean + z*eft_sd
  
  # Get dataframe
  df = data.frame(x, eft_mean, eft_sd, mean_lb, mean_ub)
  p = df %>% ggplot() +
    geom_line(aes(x, eft_mean, color = color), size = 1.1) +
    geom_ribbon(aes(x = x, ymin=mean_lb, ymax=mean_ub, fill = color), alpha=0.2) +
    theme_bw() +
    theme(axis.line = element_line(color = "grey70"),
          panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
          legend.position = 'bottom', legend.text = element_text(size = 12)) +
    labs(x = "X", y = "F(x)", title = title,color = "Weights" ,fill = "Weights") +
    scale_color_manual(values = color) + 
    scale_fill_manual(values = color) + 
    coord_cartesian(ylim = y_lim) + 
    guides(fill = 'none', color = 'none') 
  return(p)
}

# Plot mean fit with expansions
plot_mean_gg2 = function(mean_fit, ex_data,colors = color_list, line_type_list = lty_list, y_lim = c(0,5),in_labs = NULL, grid_on = FALSE){
  K = ncol(ex_data$f_test)
  df = data.frame(ex_data$x_test, ex_data$f_test, ex_data$fg_test, mean_fit)
  colnames(df) = c('x_test', paste0('f',1:K),'ftrue', "mean")
  if(is.null(in_labs)){in_labs = colnames(df)[-1]}
  df_melt = melt(df, id.vars = 'x_test')
  p = df_melt %>% ggplot() +
    geom_line(aes(x_test, value, color = variable, linetype = variable), size = 1.1) + 
    scale_color_manual(values = c(colors$exp_cols[1:K], colors$true_cols,colors$mean_cols), labels = in_labs, name = "") +
    scale_linetype_manual(values = c(line_type_list$exp_lty[1:K], line_type_list$true_lty,line_type_list$mean_lty), labels = in_labs, name = "") +
    theme_bw() +
    theme(axis.line = element_line(color = "grey70"),
          panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
          legend.position = 'bottom', legend.text = element_text(size = 12)) +
    labs(x = "X", y = "F(x)", color = "Models") +
    coord_cartesian(ylim = y_lim)
  
  if(!grid_on){
    p = p + theme(panel.grid = element_blank())
  }
  return(p)
}

# Plot prior wts
plot_prior_wts_gg2 = function(x, betax, k = 2, m = 1, ci = 0.95,y_lim = c(0,1),title = "Prior Weight Functions",inform_prior = TRUE,colors = color_list, 
                              line_type_list = lty_list, gray_scale = FALSE , w_labs = NULL){
  # Get the variance of the sum-of-trees under the informative prior  
  if(inform_prior){
    tau = 1/(2*m*k) # tau for one tree
  }else{
    tau = 1/(2*sqrt(m)*k) # tau for one tree 
  }
  tau2x = m*tau^2 # sum of variances from each tree 
  taux = sqrt(tau2x) # std of the sum-of-trees
  
  # Get upper and lower confidence bounds
  z = abs(qnorm((1-ci)/2,0,1))
  wt_lb = betax - z*taux
  wt_ub = betax + z*taux
  
  # Set column names
  K = ncol(betax)
  colnames(wt_ub) = paste0("w",1:K)
  colnames(wt_lb) = paste0("w",1:K)
  colnames(betax) = paste0("w",1:K)
  
  # Bind with x_test
  wt_mean = data.frame(x = x, betax)
  
  # Reshape dataframes for plotting
  wt_mean = reshape2::melt(wt_mean, id.vars = 'x', measure.vars = paste0("w",1:K))
  wt_lb = reshape2::melt(wt_lb)
  wt_ub = reshape2::melt(wt_ub)
  
  # Select last column in ub and lb
  wt_lb = wt_lb[,ncol(wt_lb)]
  wt_ub = wt_ub[,ncol(wt_ub)]
  
  wts_df = cbind(wt_mean, wt_lb, wt_ub)
  colnames(wts_df)[1:3] = c('x','wt','wt_mean')
  
  if(is.null(w_labs)){
    w_labs = c()
    for(j in 1:K){
      w_labs = c(w_labs,bquote(w[.(j)] * "(x)"))  
    }    
  }
  
  if(gray_scale){
    p = wts_df %>% ggplot() +
      geom_line(aes(x, wt_mean, color = wt, linetype = wt), size = 1.0) +
      geom_point(aes(x, wt_mean, color = wt), size = 1.5) +
      geom_ribbon(aes(x = x, ymin=wt_lb, ymax=wt_ub, fill = wt), alpha=0.2) +
      theme_bw() +
      theme(axis.line = element_line(color = "grey70"),
            panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
            legend.position = 'bottom', legend.text = element_text(size = 12),
            panel.grid = element_blank()) +
      labs(x = "X", y = "W(x)", title = title,color = "Weights" ,fill = "Weights", linetype = 'Weights') +
      scale_color_manual(values = colors$exp_cols[1:K], labels = w_labs) +
      scale_linetype_manual(values = line_type_list$wts_lty[1:K], labels = w_labs) +
      scale_fill_manual(values = colors$exp_cols[1:K], labels = w_labs) + 
      coord_cartesian(ylim = y_lim) + 
      guides(fill = 'none') 
  }else{
    p = wts_df %>% ggplot() +
      geom_line(aes(x, wt_mean, color = wt), size = 1.0) +
      geom_point(aes(x, wt_mean, color = wt), size = 1.5) +
      geom_ribbon(aes(x = x, ymin=wt_lb, ymax=wt_ub, fill = wt), alpha=0.2) +
      theme_bw() +
      theme(axis.line = element_line(color = "grey70"),
            panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
            legend.position = 'bottom', legend.text = element_text(size = 12)) +
      labs(x = "X", y = "W(x)", title = title,color = "Weights" ,fill = "Weights") +
      scale_color_manual(values = colors$exp_cols[1:K], labels = w_labs) + 
      scale_fill_manual(values = colors$exp_cols[1:K], labels = w_labs) + 
      coord_cartesian(ylim = y_lim)
  }
  return(p)  
}


# Plot Prior Fit 
plot_prior_fit_gg2 = function(ex_data, betax, f_true = TRUE,title = "Pointwise Prior Mean Prediction", colors = color_list, line_type_list = lty_list, y_lim = c(0,5), in_labs = NULL, grid_on = TRUE){
  # Get prior mean prediction
  prior_mean = rowSums(betax*ex_data$f_train)
  
  # Bind data and get params
  df = data.frame(x_train = ex_data$x_train, mmean = prior_mean, f_train = ex_data$f_train)
  n = nrow(df)
  K = ncol(ex_data$f_train)
  exp_cols = (ncol(df) - K+1):ncol(df)
  colnames(df)[exp_cols] = paste0('f',1:K)
  
  # Add in the true function if f_true is TRUE
  if(f_true){
    df$f_true = ex_data$fg_train
    msr_vars = c(colnames(df)[exp_cols],'f_true','mmean')
  }else{
    msr_vars = c(colnames(df)[exp_cols],'mmean')
  }
  
  # Prepare data
  exp_df_melt = melt(df, id.vars = "x_train", measure.vars = colnames(df)[exp_cols])
  df_melt = melt(df, id.vars = "x_train", measure.vars = msr_vars)
  
  # Set sizes
  line_group = c(line_type_list$exp_lty[1:K],line_type_list$true_lty,line_type_list$mean_lty)
  size_group = c(rep(1.1,(K+1)),1.2)
  color_group = c(colors$exp_cols[1:K],colors$true_cols, colors$mean_cols[1])
  
  if(is.null(in_labs)){in_labs = c(colnames(df)[exp_cols], "true", 'mean')}
  
  p = df_melt %>% ggplot() +
    geom_line(aes(x_train, value, color = variable, linetype = variable, size = variable)) + 
    scale_color_manual(values = color_group, 
                       name = "", labels = in_labs) + 
    scale_linetype_manual(values = line_group, name = "", labels = in_labs) +
    scale_size_manual(values = size_group) +
    theme_bw() +
    theme(axis.line = element_line(color = "grey70"),
          panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
          legend.position = "bottom", legend.text = element_text(size = 10)) +
    labs(x = "X", y = "F(x)", title = title) +
    coord_cartesian(ylim = y_lim) +
    guides(size = 'none')
  
  if(!grid_on){
    p = p + theme(panel.grid = element_blank())
  }
  return(p)
    
}

# Plot mu prior
plot_mu_prior = function(k = 2, m = 1, betax = 1/2, data_rng = NULL, wt_nums = c(1,2), inform_prior = FALSE,
                         x_lim = NULL, y_lim = NULL, color = 'green',title = NULL, nlevels = 4, in_labs = NULL){
  if(!inform_prior){
    prior = "Non-Informative"
    tau = 1/(2*k*sqrt(m))  
    beta_hat = rep(betax/m,2)
  }else{
    # Set parameters
    if(is.null(data_rng)){data_rng = 1:nrow(betax)}
    prior = "Informative"
    tau = 1/(2*k*m)
    beta_hat = apply(betax[data_rng,],2,mean)/m
    
    # Get the desired columns
    beta_hat = beta_hat[wt_nums]
  }
  
  #Get number of terminal node parameters
  w1 = wt_nums[1]
  w2 = wt_nums[2]
  
  #Set title indicator
  if(is.null(title)){title = "Prior for Terminal Node Parameters"}
  
  #Set ellipse info
  ellipse_list = list()
  
  #Check title
  for(l in 1:nlevels){
    #Ellipse info
    einfo = DrawEllipse(x = beta_hat[1], beta_hat[2], 
                     radius.x = l*tau,radius.y = l*tau, plot = FALSE)
    
    ellipse_list[[l]] = cbind(c(einfo$x,einfo$x[1]) , c(einfo$y,einfo$y[1]))
  }
  
  # Color scale
  if(length(color)<nlevels){
    cs =paste0(color,nlevels:1)  
  }
  
  labs = paste0(1:nlevels)
  
  # Check x and y lim 
  if(is.null(x_lim)){ x_lim=c(beta_hat[1]-(nlevels+1)*tau,beta_hat[1]+(nlevels+1)*tau) }
  if(is.null(y_lim)){ y_lim=c(beta_hat[2]-(nlevels+1)*tau,beta_hat[2]+(nlevels+1)*tau) }
  
  # x and y Labels
  if(is.null(in_labs)){
    x_lab =  bquote(mu[.(wt_nums[1])])
    y_lab =  bquote(mu[.(wt_nums[2])])
  }else{
    x_lab = in_labs[1]
    y_lab = in_labs[2]
  }
  
  #Initialize the plot
  plot(beta_hat[1],beta_hat[2], panel.first = {grid(col = 'lightgrey')}, cex = 0.8, main = title, xlab = x_lab, 
       ylab = y_lab, ylim = y_lim, xlim = x_lim, col = cs[1], pch = 16)
  for(l in 1:nlevels){
    lines(ellipse_list[[l]][,1], ellipse_list[[l]][,2], col = cs[l], lwd = 2)
  }
  legend('bottomright', title = "Std. Deviations from Mean",legend = labs, col = cs, pch = 16, cex = 0.8, ncol = 4)
  #legend('bottomright', legend = c('Non-Informative', 'Informative'), col = c('grey63', 'green'), pch = 16, cex = 0.7)
}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2D-Plots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Residuals 2D Heat Map
plot_residuals_hm_gg2 = function(x_test, resid, xcols = c(1,2), title=NULL, scale_colors = c("navy","white","darkred"),
                                 scale_vals = c(-2,0,2)){
  hm_data = data.frame(x_test, Residuals = resid)
  x1name = paste0('X',xcols[1])
  x2name = paste0('X',xcols[2])
  colnames(hm_data)[1:2] = c('x1', 'x2')
  
  if(is.null(title)){title = paste("Residual Heat Map:",x1name,"vs.",x2name)}
  p = ggplot(hm_data, aes(x1,x2,fill=Residuals)) + 
    geom_tile() + 
    labs(x = x1name, y = x2name, title = title) + 
    theme_bw() +
    theme(axis.line = element_line(color = "grey70"),
          panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
          legend.position = "right", legend.text = element_text(size = 10)) + 
    scale_fill_gradient2(low = scale_colors[1], high = scale_colors[3], mid = scale_colors[2], 
                         midpoint = scale_vals[2], limits = scale_vals[c(1,3)])
  return(p)
}

# Residuals 2D Heat Map
plot_wts_2d_gg2 = function(x_test,wmean,wnum = 1,xcols = c(1,2), title=NULL, scale_colors = c("navy","white","darkred"),
                                 scale_vals = c(-0.1,0.5,1.1)){
  wt_data = data.frame(x_test, w = wmean[,wnum])
  x1name = paste0('X',xcols[1])
  x2name = paste0('X',xcols[2])
  colnames(wt_data)[c(1,2)] = c('x1', 'x2')
  wname = paste0("W",wnum,"(x)")
  
  if(is.null(title)){title = paste0("W",wnum,"(",x1name,") vs. W",wnum,"(",x2name,")")}
  p = ggplot(wt_data, aes(x1,x2,fill=w)) + 
    geom_tile() + 
    labs(x = x1name, y = x2name, title = title, fill = wname)+ 
    theme_bw() +
    theme(axis.line = element_line(color = "grey70"),
          panel.border = element_blank(),plot.title = element_text(hjust = 0.5),
          legend.position = "right", legend.text = element_text(size = 10)) + 
    scale_fill_gradient2(low = scale_colors[1], high = scale_colors[3], mid = scale_colors[2], 
                         midpoint = scale_vals[2], limits = scale_vals[c(1,3)])
  return(p)

}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GGplot Helpers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract legend from ggplot graph
g_legend = function(gplt){
  temp = ggplot_gtable(ggplot_build(gplt))
  leg = which(sapply(temp$grobs, function(x) x$name == "guide-box"))
  legend = temp$grobs[[leg]]
  return(legend)
}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Priors
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
get_betax = function(sd_matrix){
  # Get precision matrix
  prec_matrix = 1/sd_matrix^2
  
  # Normalize the weights to be within a simplex  
  betax = prec_matrix/rowSums(prec_matrix)
  
  # Return the beta(x) matrix for the observations in the data_rng
  return(betax)
}



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Predictions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Get posterior predictive densities 
model_mix_predict_y = function(mean_draws, sd_draws){
  #Cast sd_draws to matrix
  sd_draws = as.matrix(sd_draws)
  
  #Initialize output matrix
  Npost = nrow(mean_draws)
  n = ncol(mean_draws)
  pred_dist = matrix(0, nrow = Npost, ncol = n)
  
  #Get predictions
  for(i in 1:Npost){
    pred_dist[i,] = rnorm(n, mean = mean_draws[i,], sd = sd_draws[i,])
  }
  
  return(pred_dist)
}
