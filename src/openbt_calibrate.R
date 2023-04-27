#-----------------------------------------------------
# Open BT Calibration interface
#-----------------------------------------------------
# Calibration helper to create a prior object
setprior.openbtcal = function(
    dist = c('uniform', 'normal', 'gamma'),
    mean = NULL,sd = NULL,
    a = NULL,b = NULL,
    shape = NULL,rate = NULL)
{
  # Check arguments
  valid_priors = c('uniform', 'normal', 'gamma')
  if(!(dist %in% valid_priors)){
    stop(cat("Invalid prior distribution. Valid priors include: ", paste(valid_priors,sep = ', ')))
  }
    
  if(dist == 'uniform'){
    if(is.null(a) | is.null(b)){
      stop("Invalid Uniform Prior: please enter values for a and b.")  
    }
    out = list(prior = dist, param1 = a, param2 = b)
  }  
  
  if(dist == 'normal'){
    if(is.null(mean) | is.null(sd)){
      stop("Invalid Normal Prior: please enter values for mean and sd.")  
    }
    out = list(prior = dist, param1 = mean, param2 = sd)
  } 
  
  if(dist == 'gamma'){
    if(is.null(shape) | is.null(rate)){
      stop("Invalid Gamma Prior: please enter values for shape and rate.")  
    }
    out = list(prior = dist, param1 = shape, param2 = rate)
  }  
  
  # Declare the out object to be of class openbt_prior
  class(out) = 'OpenBT_prior'
  return(out)
}

#-----------------------------------------------------
# Calibration Interface
openbtcal = function(
  yf_train, xf_train,
  yc_train, xc_train,
  ucols = c(),
  mu1 = NULL, mu2 = NULL, 
  k1 = 1.0, k2 = 1.0,
  ntree = NULL, ntreeh=NULL,
  ndpost=1000, nskip=100,
  nadapt=1000, adaptevery=100,
  power=2.0, base=.95, tc=2,
  prior_list = NULL,
  proposal_type = "default",
  grad_stepsize = NULL,
  sigmav=rep(1,length(y_train)),
  overallsdf = NULL, overallnuf= NULL,
  overallsdc = NULL, overallnuc= NULL,
  chv = cor(xc_train,method="spearman"),
  pbd=.7, pb=.5,
  stepwpert=.1, probchv=.1,
  minnumbot=5,
  printevery=100,
  numcut=100,
  xicuts=NULL,
  summarystats=FALSE,
  model=NULL,
  modelname="model"
)
{
  #--------------------------------------------------
  # model type definitions
  modeltype=0 # undefined
  MODEL_OSBART=1 # On-site BART Calibration
  if(is.null(model)){ 
    cat("Model type not specified.\n")
    cat("Available options are:\n")
    cat("model='osbart'\n")
    stop("missing model type.\n")
  }
  if(model=="osbart"){
    modeltype=MODEL_OSBART
    if(is.null(ntree)) ntree=100
    if(is.null(ntreeh)) ntreeh=1
    if(is.null(k1)) k1=2
    if(is.null(k2)) k2=2
    if(is.null(overallsdf)) overallsdf=sd(yf_train)
    if(is.null(overallsdc)) overallsdc=sd(yc_train)
    if(is.null(overallnuf)) overallnuf=10
    if(is.null(overallnuc)) overallnuc=10
    pbd=c(pbd,0.0)
  }
  
  #--------------------------------------------------
  # Calibration priors
  #--------------------------------------------------
  # Flatten the prior objects in prior_list into a vector
  pu = length(prior_list) 
  prior_info = c()
  uhat = 0
  for(i in 1:pu){
    if(class(prior_list[[i]]) != 'OpenBT_prior'){
      stop(paste("Prior",i,"is not of type OpenBT_prior. Please create the prior using the function setprior.calibrate."))  
    }
    prior_info = c(prior_info, unlist(prior_list[[i]]))
    
    if(prior_list[[i]]$prior == 'normal'){
      uhat = prior_list[[i]]$param1
    }else if(prior_list[[i]]$prior == 'uniform'){
      uhat = (prior_list[[i]]$param1 + prior_list[[i]]$param2)/2
    }
  }
  
  #--------------------------------------------------
  # Proposal Types
  #--------------------------------------------------
  # Check type
  valid_proposals = c("default", "mala")
  if(is.null(proposal_type)){
    proposal_type = "default"
  }else{
    if(!(proposal_type %in% valid_proposals)){stop(cat("Invalid proposal type! Valid options include:",valid_proposals,sep="\n")) }  
  }
  
  #--------------------------------------------------
  # Data and MCMC arguments
  #--------------------------------------------------
  # MCMC info
  nd = ndpost
  burn = nskip
  m = ntree
  mh = ntreeh
  
  # Data info
  nf = length(yf_train)
  nc = length(yc_train)
  n = nf + nc
  px = ncol(xf_train)
  p = px + pu
  
  # Initialize the calibration parameters in xf_train data
  uf_train = matrix(uhat, nrow = nf, ncol = pu, byrow = TRUE)
  xf_train = cbind(xf_train, uf_train)
  
  # Realign calibration parameter columns to be at the end of the matrix if needed
  xcols = setdiff(1:p,ucols)
  out_ucols = seq(px,p-1,by=1) # col index for cpp, index starts at 0 
  xc_train = as.matrix(xc_train[,c(xcols,ucols)]) 
  
  # Change column names
  colnames(xc_train) = c(paste0('x',xcols),paste0('u',ucols-px))
  colnames(xf_train) = c(paste0('x',xcols),paste0('u',ucols-px))
  
  # Get mean of field data and model runs
  yf_mean = mean(yf_train)
  yc_mean = mean(yc_train)
  
  #--------------------------------------------------
  # Priors
  #--------------------------------------------------
  # Theta priors -- need to adjust calibration
  rgyf = range(yf_train)
  rgyc = range(yc_train)
  disc = 2*abs(mean(yf_train) - mean(yc_train)) 
  tau1 =  (rgyf[2] - rgyf[1])/(2*sqrt(m)*k1)
  tau2 =  disc/(sqrt(m)*k2)
  #tau2 =  disc/(2*sqrt(m)*k2)
  
  # Variance prior
  overalllambdaf = overallsdf^2
  overalllambdac = overallsdc^2

  # Tree Prior 
  powerh=power
  baseh=base
  if(length(power)>1) {
    powerh=power[2]
    power=power[1]
  }
  if(length(base)>1) {
    baseh=base[2]
    base=base[1]
  }
  
  #--------------------------------------------------
  # Tree control arguments
  #--------------------------------------------------
  # Probability of birth and death move
  pbdh=pbd
  pbh=pb
  if(length(pbd)>1) {
    pbdh=pbdh[2]
    pbd=pbd[1]
  }
  # Probability of birth
  if(length(pb)>1) {
    pbh=pb[2]
    pb=pb[1]
  }
  # Probability of perturbation
  stepwperth=stepwpert
  if(length(stepwpert)>1) {
    stepwperth=stepwpert[2]
    stepwpert=stepwpert[1]
  }
  # Probability of change of variables move
  probchvh=probchv
  if(length(probchv)>1) {
    probchvh=probchv[2]
    probchv=probchv[1]
  }
  # Min node size control
  minnumboth=minnumbot
  if(length(minnumbot)>1) {
    minnumboth=minnumbot[2]
    minnumbot=minnumbot[1]
  }
  
  #--------------------------------------------------
  # cutpoints
  #--------------------------------------------------
  # Set gradient stepsize 
  gradh = 0
  if(is.null(grad_stepsize)){
    grad_stepsize = numcut/100  
  }
  
  # Set cutpoints
  if(!is.null(xicuts)){
    # use xicuts
    xi=xicuts
  }else{
    # default to equal numcut per dimension
    x = as.matrix(rbind(xf_train, xc_train))
    x = t(x)
    xi=vector("list",p)
    minx=floor(apply(x,1,min))
    maxx=ceiling(apply(x,1,max))
    for(i in 1:p){
      xinc=(maxx[i]-minx[i])/(numcut+1)
      xi[[i]]=(1:numcut)*xinc+minx[i]
      if(i>px){gradh[i-px] = xinc*grad_stepsize}
    }
  }
  
  #--------------------------------------------------
  # Banner prints
  #--------------------------------------------------
  if(modeltype==MODEL_OSBART)
  {
    cat("Model: On-Site Bayesian Additive Regression Trees calibration model (OSBART)\n")
  }
  
  #--------------------------------------------------
  #write out config file
  #--------------------------------------------------
  xroot="x"
  yroot="y"
  sroot="s"
  chgvroot="chgv"
  hroot="h"
  xiroot="xi"

  # Create temp folder
  folder=tempdir(check=TRUE)
  if(!dir.exists(folder)) dir.create(folder)
  tmpsubfolder=tempfile(tmpdir="")
  tmpsubfolder=substr(tmpsubfolder,5,nchar(tmpsubfolder))
  tmpsubfolder=paste("openbt",tmpsubfolder,sep="")
  folder=paste(folder,"/",tmpsubfolder,sep="")
  if(!dir.exists(folder)) dir.create(folder)
  fout=file(paste(folder,"/config",sep=""),"w")
  
  # Write config lines -- figure out
  writeLines(c(paste(modeltype),xroot,yroot,sroot,chgvroot,hroot,
               paste(yf_mean),paste(yc_mean),paste(m),paste(mh),
               paste(nd),paste(burn),paste(nadapt),paste(adaptevery),
               paste(mu1),paste(tau1),paste(mu2),paste(tau2),
               paste(overalllambdaf),paste(overalllambdac),paste(overallnuf),paste(overallnuc),
               paste(base),paste(power),paste(baseh),paste(powerh),paste(pu),paste(out_ucols),
               proposal_type,paste(prior_info),paste(gradh),paste(tc),
               paste(pbd),paste(pb),paste(pbdh),paste(pbh),paste(stepwpert),paste(stepwperth),
               paste(probchv),paste(probchvh),paste(minnumbot),paste(minnumboth),
               paste(printevery),paste(xiroot),paste(modelname),paste(summarystats)),fout)
  close(fout)
  
  #--------------------------------------------------
  # write out data subsets
  #--------------------------------------------------
  nslv=tc-1
  yflist=split(yf_train,(seq(nf)-1) %/% (nf/nslv))
  yclist=split(yc_train,(seq(nc)-1) %/% (nc/nslv))
  for(i in 1:nslv) write(c(yflist[[i]],yclist[[i]]), file=paste(folder,"/",yroot,i,sep=""))
  
  xflist=split(as.data.frame(xf_train),(seq(nf)-1) %/% (nf/nslv))
  xclist=split(as.data.frame(xc_train),(seq(nc)-1) %/% (nc/nslv))
  for(i in 1:nslv) write(t(rbind(xflist[[i]],xclist[[i]])),file=paste(folder,"/",xroot,i,sep=""))
  
  slist=split(sigmav,(seq(n)-1) %/% (n/nslv))
  for(i in 1:nslv) write(slist[[i]],file=paste(folder,"/",sroot,i,sep=""))
  
  chv[is.na(chv)]=0 # if a var as 0 levels it will have a cor of NA so we'll just set those to 0.
  write(chv,file=paste(folder,"/",chgvroot,sep=""))
  
  for(i in 1:p) write(xi[[i]],file=paste(folder,"/",xiroot,i,sep=""))
  
  # Set the design matrix
  hlist = vector('list',nslv)
  for(i in 1:nslv){
    nfslv = length(yflist[[i]])
    ncslv = length(yclist[[i]])
    hlist[[i]] = cbind(rep(1,nfslv+ncslv), c(rep(1,nfslv),rep(0,ncslv)))
    hlist[[i]] = as.data.frame(hlist[[i]])
  }
  for(i in 1:nslv) write(t(hlist[[i]]),file=paste(folder,"/",hroot,i,sep=""))

  rm(chv)
  #--------------------------------------------------
  # run program
  #--------------------------------------------------
  cmdopt=100 #default to serial/OpenMP
  runlocal=FALSE
  cmd="openbtcalibrate --conf"
  if(Sys.which("openbtcalibrate")[[1]]=="") # not installed in a global location, so assume current directory
    runlocal=TRUE
  
  if(runlocal) cmd="./openbtcli --conf"
  
  cmdopt=system(cmd)
  
  if(cmdopt==101) # MPI
  {
    cmd=paste("mpirun -np ",tc," openbtcalibrate ",folder,sep="")
  }
  
  if(cmdopt==100)  # serial/OpenMP
  { 
    if(runlocal)
      cmd=paste("./openbtcalibrate ",folder,sep="")
    else
      cmd=paste("openbtcalibrate ",folder,sep="")
  }
  
  #cat(cmd)
  system(cmd)
  #system(paste("rm -f ",folder,"/config",sep=""))
  #system(paste("mv ",folder,"fit ",folder,modelname,".fit",sep=""))
  
  # Format output of the function
  res=list()
  res$modeltype=modeltype; res$model=model
  res$nf = nf; res$nc = nc;res$px = px; res$pu = pu;
  res$prior_list = prior_list; res$xcols = xcols; res$ucols = ucols;
  res$xroot=xroot; res$yroot=yroot;res$m=m; res$mh=mh; res$nd=nd; res$burn=burn
  res$nadapt=nadapt; res$adaptevery=adaptevery;res$mu1=mu1;res$mu2=mu2;res$tau1=tau1;res$tau2=tau2;
  res$overalllambdaf=overalllambdaf; res$overallnuf=overallnuf;
  res$overalllambdaf=overalllambdac; res$overallnuf=overallnuc;
  res$base=base; res$power=power; res$baseh=baseh; res$powerh=powerh
  res$proposal_type = proposal_type; res$gradstep = grad_stepsize;
  res$tc=tc; res$sroot=sroot; res$chgvroot=chgvroot;
  res$pbd=pbd; res$pb=pb; res$pbdh=pbdh; res$pbh=pbh; res$stepwpert=stepwpert; res$stepwperth=stepwperth
  res$probchv=probchv; res$probchvh=probchvh; res$minnumbot=minnumbot; res$minnumboth=minnumboth
  res$printevery=printevery; res$xiroot=xiroot; res$minx=minx; res$maxx=maxx;
  res$summarystats=summarystats; res$modelname=modelname
  class(xi)="OpenBT_cutinfo"
  res$xicuts=xi
  res$folder=folder
  class(res)="OpenBT_posterior"
  
  return(res)
}

# Prediction
predict.openbtcal = function(fit=NULL,xf_test=NULL,xc_test=NULL,ucols=NULL,tc=2,q.lower=0.025,q.upper=0.975){
  # model type definitions
  MODEL_OSBART=1

  #--------------------------------------------------
  # params
  if(is.null(fit)) stop("No fitted model specified!\n")
  if(is.null(xf_test)) stop("No prediction points specified!\n")

  if(is.null(xc_test)) cat("Getting predictions of the true system using x_test and the posterior draws of u...")
  if(!is.null(xc_test)){ 
    cat(paste("Getting predictions of the true system using xf_test and the posterior draws of u.\n",
    "Getting predictions from the emulator using xc_test..."))
  }
  
  # Get data info
  xf_test=as.matrix(xf_test)
  xc_test=as.matrix(xc_test)

  px = ncol(xf_test)
  pu = length(ucols)
  p = px + pu
  nf = nrow(xf_test)
  nc = ifelse(is.null(nrow(xc_test)),0,nrow(xc_test))
  
  # organize columns
  xcols = setdiff(1:p,ucols)
  out_ucols = seq(px,p-1,by=1) # col index for cpp, index starts at 0 
  if(!is.null(xc_test)){
    xc_test = as.matrix(xc_test[,c(xcols,ucols)])
    colnames(xc_test) = c(paste0('x',xcols),paste0('u',ucols-px))
  }
  
  #yf_mean = fit$yf_mean
  #yc_mean = fit$yc_mean
  yf_mean = yc_mean = 0
  
  # Set roots
  xproot="xp"
  fproot="hp"
  folder = fit$folder

  # Set initial value for calibration parameter (will overriden in cpp so this is just a placeholder)
  uf_test = matrix(1, nrow = nf, ncol = pu)
  xf_test = cbind(xf_test, uf_test)
  
  # Change column names
  colnames(xc_test) = c(paste0('x',xcols),paste0('u',ucols-px))
  colnames(xf_test) = c(paste0('x',xcols),paste0('u',ucols-px))
  
  #--------------------------------------------------
  #write out config file
  fout=file(paste(fit$folder,"/config.calibratepred",sep=""),"w")
  writeLines(c(fit$modelname,fit$modeltype,fit$xiroot,xproot,fproot,
               paste(fit$nd),paste(fit$m),
               paste(fit$mh),paste(p),paste(pu),paste(tc),
               paste(yf_mean),paste(yc_mean),paste(out_ucols)), fout)
  close(fout)
  
  #--------------------------------------------------
  #write out data subsets
  nslv = tc

  # xdata and design matrix
  nlist = vector("list",nslv)
  if(nc>0){
    # Set the x's
    xflist=split(as.data.frame(xf_test),(seq(nf)-1) %/% (nf/nslv))
    xclist=split(as.data.frame(xc_test),(seq(nc)-1) %/% (nc/nslv))
    for(i in 1:nslv) write(t(rbind(xflist[[i]],xclist[[i]])),file=paste(folder,"/",xproot,i-1,sep=""))
      
    # Set the design matrix
    hlist = vector('list',nslv)
    for(i in 1:nslv){
      nfslv = nrow(xflist[[i]])
      ncslv = nrow(xclist[[i]])
      hlist[[i]] = cbind(rep(1,nfslv+ncslv), c(rep(1,nfslv),rep(0,ncslv)))
      hlist[[i]] = as.data.frame(hlist[[i]])
      nlist[[i]] = c(nfslv, ncslv)
    }
    for(i in 1:nslv) write(t(hlist[[i]]),file=paste(folder,"/",fproot,i-1,sep=""))
  }else{
    # Set the x's
    xlist=split(as.data.frame(xf_test),(seq(n)-1) %/% (nf/nslv))
    for(i in 1:nslv) write(t(xlist[[i]]),file=paste(fit$folder,"/",xproot,i-1,sep=""))
    
    # Set the design matrix
    hlist = vector('list',nslv)
    for(i in 1:nslv){
      nfslv = nrow(xlist[[i]])
      hlist[[i]] = as.data.frame(matrix(1,nrow=nfslv,ncol=2))
    }
    for(i in 1:nslv) write(t(hlist[[i]]),file=paste(folder,"/",fproot,i-1,sep=""))
  }

  # Cutpoints
  for(i in 1:p) write(fit$xicuts[[i]],file=paste(fit$folder,"/",fit$xiroot,i,sep=""))
  
  #--------------------------------------------------
  #run prediction program
  cmdopt=100 #default to serial/OpenMP
  runlocal=FALSE
  cmd="openbtcli --conf"
  if(Sys.which("openbtcli")[[1]]=="") # not installed in a global location, so assume current directory
    runlocal=TRUE
  
  if(runlocal) cmd="./openbtcli --conf"
  
  cmdopt=system(cmd)
  
  if(cmdopt==101){
    cmd=paste("mpirun -np ",tc," openbtcalibratepred ",fit$folder,sep="")
  }
  
  if(cmdopt==100){ 
    if(runlocal)
      cmd=paste("./openbtcalibratepred ",fit$folder,sep="")
    else
      cmd=paste("openbtcalibratepred ",fit$folder,sep="")
  }
  
  system(cmd)
  #system(paste("rm -f ",fit$folder,"/config.calibratepred",sep=""))
  
  #--------------------------------------------------
  #format and return
  res=list()
  
  fnames=list.files(fit$folder,pattern=paste(fit$modelname,".etadraws*",sep=""),full.names=TRUE)
  res$etadraws=do.call(cbind,sapply(fnames,data.table::fread))
  
  fnames=list.files(fit$folder,pattern=paste(fit$modelname,".deltadraws*",sep=""),full.names=TRUE)
  res$deltadraws=do.call(cbind,sapply(fnames,data.table::fread))

  #fnames=list.files(fit$folder,pattern=paste(fit$modelname,".mdraws*",sep=""),full.names=TRUE)
  #res$mdraws=do.call(cbind,sapply(fnames,data.table::fread))
  
  fnames=list.files(fit$folder,pattern=paste(fit$modelname,".sfdraws*",sep=""),full.names=TRUE)
  res$sfdraws=do.call(cbind,sapply(fnames,data.table::fread))
  
  if(nc>0){
    fnames=list.files(fit$folder,pattern=paste(fit$modelname,".scdraws*",sep=""),full.names=TRUE)
    res$scdraws=do.call(cbind,sapply(fnames,data.table::fread))
    
    # reorder results to stack xf_test then xc_test
    nf_order = c()
    nc_order = c()
    ntot = 0
    for(l in 1:nslv){
      nf_order = c(nf_order, 1:nlist[[l]][1] + ntot)
      nc_order = c(nc_order, 1:nlist[[l]][2] + nlist[[l]][1] + ntot)
      ntot = ntot + sum(nlist[[l]])
    } 
    res$etadraws = res$etadraws[,c(nf_order,nc_order)] 
    res$deltadraws = res$deltadraws[,c(nf_order,nc_order)] 

    rownames(res$etadraws) = NULL
    rownames(res$deltadraws) = NULL        
  }
  
  # Store Results
  # Eta Posterior
  res$etamean=apply(res$etadraws,2,mean)
  res$etasd=apply(res$etadraws,2,sd)
  res$eta.5=apply(res$etadraws,2,quantile,0.5)
  res$eta.lower=apply(res$etadraws,2,quantile,q.lower)
  res$eta.upper=apply(res$etadraws,2,quantile,q.upper)
   
  # delta Posterior
  res$deltamean=apply(res$deltadraws,2,mean)
  res$deltasd=apply(res$deltadraws,2,sd)
  res$delta.5=apply(res$deltadraws,2,quantile,0.5)
  res$delta.lower=apply(res$deltadraws,2,quantile,q.lower)
  res$delta.upper=apply(res$deltadraws,2,quantile,q.upper)

  # f = eta + delta posterior -- only for field obs
  res$fdraws = res$etadraws[,1:nf] + res$deltadraws[,1:nf]
  res$fmean=apply(res$fdraws,2,mean)
  res$fsd=apply(res$fdraws,2,sd)
  res$f.5=apply(res$fdraws,2,quantile,0.5)
  res$f.lower=apply(res$fdraws,2,quantile,q.lower)
  res$f.upper=apply(res$fdraws,2,quantile,q.upper)
  
  # Mean draws (remove)
  # res$mmean=apply(res$mdraws,2,mean)
  # res$msd=apply(res$mdraws,2,sd)
  # res$m.5=apply(res$mdraws,2,quantile,0.5)
  # res$m.lower=apply(res$mdraws,2,quantile,q.lower)
  # res$m.upper=apply(res$mdraws,2,quantile,q.upper)
  
  # Field data Error std Posterior
  res$sfmean=apply(res$sfdraws,2,mean)
  res$sfsd=apply(res$sfdraws,2,sd)
  res$sf.5=apply(res$sfdraws,2,quantile,0.5)
  res$sf.lower=apply(res$sfdraws,2,quantile,q.lower)
  res$sf.upper=apply(res$sfdraws,2,quantile,q.upper)
  
  # Model runs Error std Posterior
  if(nc>0){
    res$scmean=apply(res$scdraws,2,mean)
    res$scsd=apply(res$scdraws,2,sd)
    res$sc.5=apply(res$scdraws,2,quantile,0.5)
    res$sc.lower=apply(res$scdraws,2,quantile,q.lower)
    res$sc.upper=apply(res$scdraws,2,quantile,q.upper)
  }else{
    res$scmean=NULL
    res$scsd=NULL
    res$sc.5=NULL
    res$sc.lower=NULL
    res$sc.upper=NULL
  }
  
  # Other quantities  
  res$q.lower=q.lower
  res$q.upper=q.upper
  res$modeltype=fit$modeltype
  
  class(res)="OpenBT_predict"
  
  return(res)
}

# Get Posterior of calibration parameters
posterior.openbtcal = function(fit,q.lower=0.025,q.upper=0.975){
  if(is.null(fit)) stop("No fitted model specified!\n")
  pu = fit$pu
  res = list()
  fnames=list.files(fit$folder,pattern=paste(fit$modelname,".udraws*",sep=""),full.names=TRUE)
  
  res$udraws=do.call(cbind,sapply(fnames,data.table::fread))  
  
  # Remove the first row
  res$udraws=matrix(res$udraws[-c(1:pu),], ncol = pu)
  
  # Summary stats
  res$umean = apply(res$udraws,2,mean)
  res$usd = apply(res$udraws,2,sd)
  res$u.5 = apply(res$udraws,2,median)
  res$u.lower = apply(res$udraws,2,quantile, q.lower)
  res$u.upper = apply(res$udraws,2,quantile, q.upper)
  
  res$q.lower=q.lower
  res$q.upper=q.upper
  res$modeltype=fit$modeltype
  
  class(res)="OpenBT_calparams"
  return(res)
}
