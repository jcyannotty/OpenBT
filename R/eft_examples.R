#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Name: eft_examples.R
# Author: John Yannotty (yannotty.1@buckeyemail.osu.edu)
# Desc: R code to perform model mixing using BART on three EFT examples.
#       The tuning parameters that are currently set were used to produce the plots,
#       which appear in the file "EFT-Model-Mixing-Using-BART.html" 
# Version: 1.0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### SETUP TUTORIAL LIKE THINGS
setwd("/home/johnyannotty/Documents/Open BT Project SRC")
source("/home/johnyannotty/Documents/Open BT Project SRC/openbt.R")
source('/home/johnyannotty/Documents/Model Mixing BART/Model Mixing R Code/physics expansion r functions.R')
source('/home/johnyannotty/Documents/Model Mixing BART/Model Mixing R Code/Model Mixing Helper Functions.R')

#----------------------------------------------------------
#Get training data
ex1_data = get_data(20, 300, 0.005, 2, 4, 0.03, 0.5, 321)
ex2_data = get_data(20, 300, 0.005, 4, 4, 0.03, 0.5, 321)
ex3_data = get_data(20, 300, 0.005, c(2,4,6), NULL, 0.03, 0.5, 321)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot Expansion
labs1 = c(expression(paste(f[s]^"(2)", '(x)')), expression(paste(f[l]^"(4)", '(x)')), expression(paste(f["\u2020"],'(x)')))
e1 = plot_exp_gg2(ex1_data, in_labs = labs1, colors = color_list, line_type_list = lty_list, grid_on = TRUE) 
e1 = e1 + theme(legend.position = c(0.22,0.16),legend.title = element_blank()) + labs(title = '(a)')
e1 = e1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), plot.title = element_text(size = 15),legend.text.align = 0, legend.text = element_text(size = 13))

labs2 = c(expression(paste(f[s]^"(4)", '(x)')), expression(paste(f[l]^"(4)", '(x)')), expression(paste(f["\u2020"],'(x)')))
e2 = plot_exp_gg2(ex2_data, in_labs = labs2, colors = color_list, line_type_list = lty_list, grid_on = TRUE) 
e2 = e2 + theme(legend.position = c(0.22,0.16),legend.title = element_blank()) + labs(title = '(b)')
e2 = e2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), plot.title = element_text(size = 15),legend.text.align = 0, legend.text = element_text(size = 13))

labs3 = c(expression(paste(f[s]^"(2)", '(x)')), expression(paste(f[s]^"(4)", '(x)')),expression(paste(f[s]^"(6)", '(x)')), expression(paste(f["\u2020"],'(x)')))
e3 = plot_exp_gg2(ex3_data, in_labs = labs3, colors = color_list, line_type_list = lty_list, grid_on = TRUE) 
e3 = e3 + theme(legend.position = c(0.22,0.19),legend.title = element_blank()) + labs(title = '(c)')
e3 = e3 + theme(axis.text=element_text(size=12),axis.title=element_text(size=13), plot.title = element_text(size = 15),legend.text.align = 0, legend.text = element_text(size = 12.5))

e1 = e1 + theme(legend.key.width = unit(0.73,"cm")) + guides(linetype = guide_legend(override.aes = list(size = 0.7)))
e2 = e2 + theme(legend.key.width = unit(0.73,"cm")) + guides(linetype = guide_legend(override.aes = list(size = 0.7)))
e3 = e3 + theme(legend.key.width = unit(0.73,"cm")) + guides(linetype = guide_legend(override.aes = list(size = 0.7)))

grid.arrange(arrangeGrob(e1,e2,e3,nrow = 1), nrow=1, heights = c(8), 
             top=textGrob("Prototype Effective Field Theories", gp=gpar(fontsize=20,font=8)))

#-----------------------------------------------------
# Ex1 Non-Informative Prior
#-----------------------------------------------------
# Attach the data set
attach(ex1_data)

# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit1=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),model="mixbart",
           ntree = 8,k = 2.5, overallsd = sqrt(sig2_hat), overallnu = 5,power = 2.0, base = 0.95,
           ntreeh=1,numcut=300,tc=4,minnumbot = 4,
           ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 1000,
           summarystats = FALSE,modelname="eft_mixing")

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp1=predict.openbt(fit1,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw1=openbt.mixingwts(fit1, x.test = x_test, numwts = 2, tc = 4, q.lower = 0.025, q.upper = 0.975)

detach(ex1_data)

#-----------------------------------------------------
# Ex1 Informative Prior
#-----------------------------------------------------
# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit2=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),f.sd.train = f_train_dsd,model="mixbart",
           ntree = 10,k = 5, overallsd = sqrt(sig2_hat), overallnu = 8, power = 2.0, base = 0.95, 
           ntreeh=1,numcut=300,tc=4,minnumbot = 3,
           ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 500,
           modelname="eft_mixing"
          )

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp2=predict.openbt(fit2,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw2 = openbt.mixingwts(fit2, x.test = x_test, numwts = 2, tc = 4, q.lower = 0.025, q.upper = 0.975)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Ex 1 Plot the predictions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set the labels for hte legend
g_labs = c(expression(paste(hat(f)[1], '(x)')), expression(paste(hat(f)[2], '(x)')),
           expression(paste(f["\u2020"],'(x)')),"Post. Mean")
# Plot the predictions obtained with the non-informative prior
p1 = plot_fit_gg2(ex1_data, fitp1, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.85,2.75), title = "Non-Informative Prior", grid_on = TRUE)
# Plot the predictions obtained with the informative prior
p2 = plot_fit_gg2(ex1_data, fitp2, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.85,2.75), title = "Informative Prior", grid_on = TRUE)
# Resize text elements
p1 = p1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
p2 = p2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure
legend1 = g_legend(p1)
grid.arrange(arrangeGrob(p1 + theme(legend.position = "none"),
                         p2 + theme(legend.position = "none"),
                         nrow = 1), nrow=2, heights = c(10,1), legend = legend1,
                         top = textGrob("Posterior Mean Predictions", gp = gpar(fontsize = 16)))

#------------------------------------------------
# Ex 1 Plot the weight functions
#------------------------------------------------
# Create the plot for weights under the non-informative prior
w1 = plot_wts_gg2(fitw1, ex1_data$x_test, y_lim = c(-0.05, 1.05),title = 'Non-Informative Prior', colors = color_list,
                  line_type_list = lty_list, gray_scale = FALSE)
# Create the plot for weights under the informative prior
w2 = plot_wts_gg2(fitw2, ex1_data$x_test, y_lim = c(-0.05, 1.05), title = 'Informative Prior',colors = color_list,
                  line_type_list = lty_list, gray_scale = FALSE)
# Resize text elements
w1 = w1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
w2 = w2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots 
legend_w1 = g_legend(w1)
grid.arrange(arrangeGrob(w1 + theme(legend.position = "none"),
                         w2 + theme(legend.position = "none"),
                         nrow = 1),nrow=2, heights = c(10,1), legend = legend_w1,
                        top = textGrob("Posterior Weight Functions", gp = gpar(fontsize = 16)))



#-----------------------------------------------------
# Ex2 Non-Informative Prior
#-----------------------------------------------------
# Attach example 2 data
attach(ex2_data)

# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit1=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),model="mixbart",
            ntree = 10,k = 3.5, overallsd = sqrt(sig2_hat), overallnu = 5,power = 2.0, base = 0.95,
            ntreeh=1,numcut=300,tc=4,minnumbot = 4,
            ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 1000,
            summarystats = FALSE,modelname="eft_mixing")

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp1=predict.openbt(fit1,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw1=openbt.mixingwts(fit1, x.test = x_test, numwts = 2, tc = 4, q.lower = 0.025, q.upper = 0.975)


#-----------------------------------------------------
# Ex2 - Informative Prior
#-----------------------------------------------------
# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit2=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),f.sd.train = f_train_dsd,model="mixbart",
            ntree = 10,k = 5, overallsd = sqrt(sig2_hat), overallnu = 5, power = 2.0, base = 0.95, 
            ntreeh=1,numcut=300,tc=4,minnumbot = 3,
            ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 500,
            modelname="eft_mixing"
)

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp2=predict.openbt(fit2,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw2 = openbt.mixingwts(fit2, x.test = x_test, numwts = 2, tc = 4, q.lower = 0.025, q.upper = 0.975)

# Attach example 2 data
detach(ex2_data)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Ex2 - Plot the predictions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set the labels in the legend
g_labs = c(expression(paste(f[1], '(x)')), expression(paste(f[2], '(x)')), expression(paste(f["\u2020"],'(x)')),
           "Post. Mean")
# Plot the predictions from the non-informative prior
p1 = plot_fit_gg2(ex2_data, fitp1, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.9,2.8), grid_on = TRUE, title = "Non-Informative Prior")

# Plot the predictions from the informative prior
p2 = plot_fit_gg2(ex2_data, fitp2, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.9,2.8), title = "Informative Prior", grid_on = TRUE)

# Resize the plot text
p1 = p1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
p2 = p2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots
legend1 = g_legend(p1)
grid.arrange(arrangeGrob(p1 + theme(legend.position = "none"),
                         p2 + theme(legend.position = "none"),
                         nrow = 1), nrow=2, heights = c(10,1), legend = legend1,
             top = textGrob("Posterior Mean Predictions", gp = gpar(fontsize = 16)))

#------------------------------------------------
# Ex2 - Plot the weight functions
#------------------------------------------------
# Create the plot for weights under the non-informative prior
w1 = plot_wts_gg2(fitw1, ex1_data$x_test, y_lim = c(-0.05, 1.05),title = 'Non-Informative Prior', colors = color_list,
                  line_type_list = lty_list, gray_scale = FALSE)
# Create the plot for weights under the informative prior
w2 = plot_wts_gg2(fitw2, ex1_data$x_test, y_lim = c(-0.05, 1.05), title = 'Informative Prior',colors = color_list,
                  line_type_list = lty_list, gray_scale = FALSE)
# Resize text elements
w1 = w1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
w2 = w2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots 
legend_w1 = g_legend(w1)
grid.arrange(arrangeGrob(w1 + theme(legend.position = "none"),
                         w2 + theme(legend.position = "none"),
                         nrow = 1),nrow=2, heights = c(10,1), legend = legend_w1,
             top = textGrob("Posterior Weight Functions", gp = gpar(fontsize = 16)))


#-----------------------------------------------------
# Ex3 - Non-Informative Prior
#-----------------------------------------------------
# Attach example 2 data
attach(ex3_data)

# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit1=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),model="mixbart",
            ntree = 10,k = 5.5, overallsd = sqrt(sig2_hat), overallnu = 3,power = 2.0, base = 0.95,
            ntreeh=1,numcut=300,tc=4,minnumbot = 4,
            ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 1000,
            summarystats = FALSE,modelname="eft_mixing")

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp1=predict.openbt(fit1,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw1=openbt.mixingwts(fit1, x.test = x_test, numwts = 3, tc = 4, q.lower = 0.025, q.upper = 0.975)


#-----------------------------------------------------
# Informative Prior
#-----------------------------------------------------
# Get the initial estimate of sigma^2 
sig2_hat = max(apply(apply(f_train, 2, function(x) (x-y_train)^2),2,min))

# Perform model mixing with the Non-informative prior.
# Note, f.sd.train is not specified, hence the non-informative prior is used by default.
fit2=openbt(x_train,y_train,f_train,pbd=c(0.7,0.0),f.sd.train = f_train_dsd,model="mixbart",
            ntree = 8, k = 6.5, overallsd = sqrt(sig2_hat), overallnu = 5, power = 2.0, base = 0.95, 
            ntreeh=1,numcut=300,tc=4,minnumbot = 3,
            ndpost = 30000, nskip = 2000, nadapt = 5000, adaptevery = 500, printevery = 500,
            modelname="eft_mixing"
)

# Get predictions at the test points specified by x_test with mean predictions stored in f_test.
fitp2=predict.openbt(fit2,x.test = x_test, f.test = f_test, tc=4, q.lower = 0.025, q.upper = 0.975)

# Get the weight functions from the fit model
fitw2 = openbt.mixingwts(fit2, x.test = x_test, numwts = 3, tc = 4, q.lower = 0.025, q.upper = 0.975)

# Attach example 2 data
detach(ex3_data)

#------------------------------------------------
# Ex3 - Plot the predictions
#------------------------------------------------
# Set the labels in the legend
g_labs = c(expression(paste(f[1], '(x)')), expression(paste(f[2], '(x)')),
           expression(paste(f[3], '(x)')), expression(paste(f["\u2020"],'(x)')),
           "Post. Mean")
# Generate the first plot
p1 = plot_fit_gg2(ex3_data, fitp1, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.8,2.7), title = "Non-Informative Prior", grid_on = TRUE)
# Generate the first plot
p2 = plot_fit_gg2(ex3_data, fitp2, in_labs = g_labs, colors = color_list, line_type_list = lty_list,
                  y_lim = c(1.8,2.7), title = "Informative Prior", grid_on = TRUE)
# Resize plot text
p1 = p1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
p2 = p2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots
legend1 = g_legend(p1)
grid.arrange(arrangeGrob(p1 + theme(legend.position = "none"),
                         p2 + theme(legend.position = "none"),
                         nrow = 1), nrow=2, heights = c(10,1), legend = legend1,
             top = textGrob("Posterior Mean Predictions", gp = gpar(fontsize = 16)))

#------------------------------------------------
# Ex3 - Plot the weight functions 
#------------------------------------------------
# Plot the weights from the non-informative prior
w1 = plot_wts_gg2(fitw1, ex3_data$x_test, y_lim = c(-0.05, 1.05), title = 'Non-Informative Prior', 
                  colors = color_list, line_type_list = lty_list, gray_scale = FALSE)
# Plot the weights from the informative prior
w2 = plot_wts_gg2(fitw2, ex3_data$x_test, y_lim = c(-0.05, 1.05), title = 'Informative Prior', 
                  colors = color_list, line_type_list = lty_list, gray_scale = FALSE)
# Resize text elements
w1 = w1+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))
w2 = w2+theme(axis.text=element_text(size=12),axis.title=element_text(size=13), 
              plot.title = element_text(size = 15))

# Create the figure with both plots
legend_w1 = g_legend(w1)
grid.arrange(arrangeGrob(w1 + theme(legend.position = "none"),
                         w2 + theme(legend.position = "none"),
                         nrow = 1),nrow=2, heights = c(10,1), legend = legend_w1,
             top = textGrob("Posterior Weight Functions", gp = gpar(fontsize = 16)))
