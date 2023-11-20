#------------------------------------------------
# Openbt Mixing R interface
#------------------------------------------------
# Load/Install required packages
required <- c("zip","data.table")
tbi <- required[!(required %in% installed.packages()[,"Package"])]
if(length(tbi)) {
  cat("***Installing OpenBT package dependencies***\n")
  install.packages(tbi,repos="https://cloud.r-project.org",quiet=TRUE)
}
library(zip,quietly=TRUE,warn.conflicts=FALSE)
library(data.table,quietly=TRUE,warn.conflicts=FALSE)


#------------------------------------------------
# Model Training and Predictions
#------------------------------------------------
train.openbtmixing = function(
    x.train,
    y.train,
    f.train = matrix(1,nrow = length(y.train), ncol = 2),
    ntree=NULL,
    ntreeh=NULL,
    ndpost=1000, nskip=100,
    k=NULL,
    power=2.0, base=.95, maxd = 999,
    tc=2,
    sigmav=rep(1,length(y.train)),
    f.sd.train = NULL,
    rpath = FALSE,
    gam = NULL, q = 2.0, rshp1 = 2, rshp2 = 2,
    overallsd = NULL,
    overallnu= NULL,
    chv = cor(x.train,method="spearman"),
    pbd=.7,
    pb=.5,
    stepwpert=.1,
    probchv=.1,
    minnumbot=5,
    printevery=100,
    batchsize = 100,
    numcut=100,
    xicuts=NULL,
    nadapt=1000,
    adaptevery=100,
    summarystats=FALSE,
    truncateds=NULL,
    model=NULL,
    modelname="model")
{
  # Define the model type
  MODEL_BARTBMM=10  
  
  if(is.null(model)){ 
    cat("Model type not specified.\n")
    cat("Available options are:\n")
    cat("model='bmm'\n")
    stop("missing model type.\n")
  }
  
  if(model=="mixbart"){
    modeltype=MODEL_BARTBMM
    if(is.null(ntree)) ntree=20
    if(is.null(ntreeh)) ntreeh=1
    if(is.null(k)) k=1
    if(is.null(overallsd)) overallsd=sd(y.train)
    if(is.null(overallnu)) overallnu=10
    pbd=c(pbd,0.0)
  }
  #--------------------------------------------------
  nd = ndpost
  burn = nskip
  m = ntree
  mh = ntreeh
  #--------------------------------------------------
  #data
  if(!is.matrix(x.train)){x.train = matrix(x.train, ncol = 1)}
  n = length(y.train)
  p = ncol(x.train)
  x = t(x.train)
  
  #Set mix discrepancy to FALSE if we use a different model
  eftprior = FALSE
  #Check to see if any discrepancy data has been passed into the function -- if so, we will use the discrepancy model mixing
  if(!is.null(f.sd.train)){
    eftprior = TRUE    
  }

  #--------------------------------------------------
  #cutpoints
  if(is.null(xicuts)){
    xi=vector("list",p)
    minx_temp=apply(x,1,min)
    maxx_temp=apply(x,1,max)
    
    maxx = round(maxx_temp,1) + ifelse((round(maxx_temp,1)-maxx_temp)>0,0,0.1)
    minx = round(minx_temp,1) - ifelse((minx_temp - round(minx_temp,1))>0,0,0.1)
    for(i in 1:p)
    {
      xinc=(maxx[i]-minx[i])/(numcut+1)
      xi[[i]]=(1:numcut)*xinc+minx[i]
    }
  }else{
    # Use default xcuts
    xi=xicuts
  }
  
  #--------------------------------------------------
  # Priors
  tau =  (1)/(2*sqrt(m)*k)
  beta0 = 1/(ncol(f.train)*m)
  overallsd = sqrt((overallnu+2)*overallsd^2/overallnu)
  overalllambda = overallsd^2
  if(eftprior){
    tau = 1/(2*(m)*k)
    beta0 = 1/m
  }  
  
  # Smoothing Prior
  if(rpath){
    if(is.null(gam)){gam = rshp1/(rshp1 + rshp2) }
  }else{
    gam = 1
  }
  
  # Tree prior
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
  # Proposal Distributions
  pbdh=pbd
  pbh=pb
  if(length(pbd)>1) {
    pbdh=pbdh[2]
    pbd=pbd[1]
  }
  if(length(pb)>1) {
    pbh=pb[2]
    pb=pb[1]
  }
  
  # Pert probability
  stepwperth=stepwpert
  if(length(stepwpert)>1) {
    stepwperth=stepwpert[2]
    stepwpert=stepwpert[1]
  }

  # Change of variable probability
  probchvh=probchv
  if(length(probchv)>1) {
    probchvh=probchv[2]
    probchv=probchv[1]
  }
  
  # Min node size
  minnumboth=minnumbot
  if(length(minnumbot)>1) {
    minnumboth=minnumbot[2]
    minnumbot=minnumbot[1]
  }
  
  #--------------------------------------------------
  # Banner
  if(modeltype==MODEL_BARTBMM){
    cat("Model: Model Mixing with Bayesian Additive Regression Trees\n")
  } 
  
  #--------------------------------------------------
  #write out config file
  xroot="x"
  yroot="y"
  sroot="s"
  chgvroot="chgv"
  froot="f"
  fsdroot="fsd"
  xiroot="xi"
  folder=tempdir(check=TRUE)
  if(!dir.exists(folder)) dir.create(folder)
  tmpsubfolder=tempfile(tmpdir="")
  tmpsubfolder=substr(tmpsubfolder,5,nchar(tmpsubfolder))
  tmpsubfolder=paste("openbt",tmpsubfolder,sep="")
  folder=paste(folder,"/",tmpsubfolder,sep="")
  if(!dir.exists(folder)) dir.create(folder)
  fout=file(paste(folder,"/config",sep=""),"w")
  writeLines(c(paste(modeltype),xroot,yroot,paste(m),paste(mh),paste(nd),paste(burn),
               paste(nadapt),paste(adaptevery),paste(tau),paste(beta0),paste(overalllambda),
               paste(overallnu),paste(base),paste(power),paste(baseh),paste(powerh),paste(maxd),
               paste(tc),paste(sroot),paste(chgvroot),paste(froot),paste(fsdroot),paste(eftprior),
               paste(rpath),paste(gam),paste(q),paste(rshp1),paste(rshp2), 
               paste(pbd),paste(pb),paste(pbdh),paste(pbh),paste(stepwpert),paste(stepwperth),
               paste(probchv),paste(probchvh),paste(minnumbot),paste(minnumboth),
               paste(printevery),paste(batchsize),
               paste(xiroot),paste(modelname),paste(summarystats)),fout)
  close(fout)

  #--------------------------------------------------
  #write out data subsets
  nslv=tc-1
  ylist=split(y.train,(seq(n)-1) %/% (n/nslv))
  for(i in 1:nslv) write(ylist[[i]],file=paste(folder,"/",yroot,i,sep=""))
  xlist=split(as.data.frame(x.train),(seq(n)-1) %/% (n/nslv))
  for(i in 1:nslv) write(t(xlist[[i]]),file=paste(folder,"/",xroot,i,sep=""))
  slist=split(sigmav,(seq(n)-1) %/% (n/nslv))
  for(i in 1:nslv) write(slist[[i]],file=paste(folder,"/",sroot,i,sep=""))
  chv[is.na(chv)]=0 # if a var as 0 levels it will have a cor of NA so we'll just set those to 0.
  write(chv,file=paste(folder,"/",chgvroot,sep=""))
  for(i in 1:p) write(xi[[i]],file=paste(folder,"/",xiroot,i,sep=""))
  rm(chv)
  
  #Write the function output 
  flist=split(as.data.frame(f.train),(seq(n)-1) %/% (n/nslv))
  for(i in 1:nslv) write(t(flist[[i]]),file=paste(folder,"/",froot,i,sep=""))
  
  if(eftprior){
    fdslist=split(as.data.frame(f.sd.train),(seq(n)-1) %/% (n/nslv))
    for(i in 1:nslv) write(t(fdslist[[i]]),file=paste(folder,"/",fsdroot,i,sep=""))
  }
  
  #--------------------------------------------------
  #run program
  cmdopt=100 #default to serial/OpenMP
  runlocal=FALSE
  cmd="openbtcli --conf"
  if(Sys.which("openbtcli")[[1]]=="") # not installed in a global location, so assume current directory
    runlocal=TRUE
  
  if(runlocal) cmd="./openbtcli --conf"
  
  cmdopt=system(cmd)
  
  if(cmdopt==101) # MPI
  {
    cmd=paste("mpirun -np ",tc," openbtcli ",folder,sep="")
  }
  
  if(cmdopt==100)  # serial/OpenMP
  { 
    if(runlocal)
      cmd=paste("./openbtmixingts ",folder,sep="")
    else
      cmd=paste("openbtmixingts ",folder,sep="")
  }
  
  system(cmd)
  
  res=list()
  res$modeltype=modeltype
  res$model=model
  res$xroot=xroot; res$yroot=yroot;res$m=m; res$mh=mh; res$nd=nd; res$burn=burn
  res$nadapt=nadapt; res$adaptevery=adaptevery; res$tau=tau;res$beta0=beta0;res$overalllambda=overalllambda
  res$overallnu=overallnu; res$k=k; res$base=base; res$power=power; res$baseh=baseh; res$powerh=powerh; res$maxd = maxd;
  res$tc=tc; res$sroot=sroot; res$chgvroot=chgvroot;res$froot=froot;res$fsdroot=fsdroot; 
  res$eftprior = eftprior;res$rpath = rpath; res$rshp1 = rshp1; res$rshp2 = rshp2;
  res$pbd=pbd; res$pb=pb
  res$pbdh=pbdh; res$pbh=pbh; res$stepwpert=stepwpert; res$stepwperth=stepwperth
  res$probchv=probchv; res$probchvh=probchvh; res$minnumbot=minnumbot; res$minnumboth=minnumboth
  res$printevery=printevery; res$xiroot=xiroot;
  res$summarystats=summarystats; res$modelname=modelname
  class(xi)="OpenBT_cutinfo"
  res$xicuts=xi
  res$folder=folder
  class(res)="OpenBT_posterior"
  
  return(res)
}


# Prediction function for mixed-prediction, weights, and sigma
predict.openbtmixing = function(
  fit=NULL,
  x.test=NULL,
  f.test=matrix(1,nrow = 1, ncol = 2),
  ptype = "all",
  tc=2,
  q.lower=0.025,
  q.upper=0.975,
  batchsize = 100)
{
  #--------------------------------------------------
  # params
  if(is.null(fit)) stop("No fitted model specified!\n")
  if(is.null(x.test)) stop("No prediction points specified!\n")
  if(ptype == "all"){
    domdraws = TRUE;dosdraws = TRUE;dopdraws = FALSE 
  }else if(ptype == "mean"){
    domdraws = TRUE;dosdraws = FALSE;dopdraws = FALSE
  }else if(ptype == "sigma"){
    domdraws = FALSE;dosdraws = TRUE;dopdraws = FALSE
  }else{
    cat("Prediction type not specified.\n")
    cat("Available options are:\n")
    cat("ptype='all'\n")
    cat("ptype='mean'\n")
    cat("ptype='sigma'\n")
    stop("missing model type.\n")      
  }
    
  nslv=tc
  x.test=as.matrix(x.test)
  p=ncol(x.test)
  n=nrow(x.test)
  k=2 #Default number of models for model mixing
  xproot="xp"
  fproot="fp"
  rpath = fit$rpath
  gammaroot = "rpg"
  
  if(is.null(f.test)){stop("No function output specified for model mixing!\n")}
  k=ncol(f.test) #Number of models
  
  #--------------------------------------------------
  # Batch size calculation
  nd = fit$nd
  numbatches = ceiling(nd/batchsize)
  
  #--------------------------------------------------
  #write out config file
  fout=file(paste(fit$folder,"/config.pred",sep=""),"w")
  writeLines(c(fit$modelname,fit$modeltype,fit$xiroot,xproot,fproot,
               paste(fit$nd),paste(fit$m),
               paste(fit$mh),paste(p),paste(k),
               paste(domdraws),paste(dosdraws),paste(dopdraws),
               paste(batchsize),paste(numbatches),paste(tc),
               paste(rpath), gammaroot), fout)
  close(fout)
  
  #--------------------------------------------------
  #write out data subsets
  #folder=paste(".",fit$modelname,"/",sep="")
  xlist=split(as.data.frame(x.test),(seq(n)-1) %/% (n/nslv))
  for(i in 1:nslv) write(t(xlist[[i]]),file=paste(fit$folder,"/",xproot,i-1,sep=""))
  for(i in 1:p) write(fit$xicuts[[i]],file=paste(fit$folder,"/",fit$xiroot,i,sep=""))
  
  flist=split(as.data.frame(f.test),(seq(n)-1) %/% (n/nslv))
  for(i in 1:nslv) write(t(flist[[i]]),file=paste(fit$folder,"/",fproot,i-1,sep=""))
  
  
  #--------------------------------------------------
  #run prediction program
  cmdopt=100 #default to serial/OpenMP
  runlocal=FALSE
  cmd="openbtcli --conf"
  if(Sys.which("openbtcli")[[1]]=="") # not installed in a global location, so assume current directory
    runlocal=TRUE
  
  if(runlocal) cmd="./openbtcli --conf"
  
  cmdopt=system(cmd)
  
  if(cmdopt==101) # MPI
  {
    cmd=paste("mpirun -np ",tc," openbtpred ",fit$folder,sep="")
  }
  
  if(cmdopt==100)  # serial/OpenMP
  { 
    if(runlocal)
      cmd=paste("./openbtmixingpredts ",fit$folder,sep="")
    else
      cmd=paste("openbtmixingpredts ",fit$folder,sep="")
  }
  
  #cmd=paste("mpirun -np ",tc," openbtpred",sep="")
  #cat(cmd)
  system(cmd)
  system(paste("rm -f ",fit$folder,"/config.pred",sep=""))
  
  
  #--------------------------------------------------
  #format and return
  res=list()
  
  # Faster using data.table's fread than the built-in read.table.
  # However, it does strangely introduce some small rounding error on the order of 8.9e-16.
  if(domdraws){
    fnames=list.files(fit$folder,pattern=paste(fit$modelname,".mdraws*",sep=""),full.names=TRUE)
    res$mdraws=do.call(cbind,sapply(fnames,data.table::fread))
    
    # Now return weights
    wt_list = list()
    mean_matrix = sd_matrix = ub_matrix = lb_matrix = med_matrix = matrix(0, nrow = n, ncol = numwts)
    
    #Get the file names for the model weights 
    #--file name for model weight j using processor p is ".wjdrawsp"
    for(j in 1:numwts){
      #Get the files for weight j
      tagname = paste0(".w", j,"draws*")
      fnames=list.files(fit$folder,pattern=paste(fit$modelname,tagname,sep=""),full.names=TRUE)
      
      #Bind the posteriors for weight j across all x points -- npost X n data 
      wt_list[[j]] = do.call(cbind,sapply(fnames,data.table::fread))
      
      #Now populate the summary stat matrices -- n X k matrices
      mean_matrix[,j] = apply(wt_list[[j]], 2, mean)
      sd_matrix[,j] = apply(wt_list[[j]], 2, sd)
      med_matrix[,j] = apply(wt_list[[j]], 2, median)
      lb_matrix[,j] = apply(wt_list[[j]], 2, quantile,q.lower)
      ub_matrix[,j] = apply(wt_list[[j]], 2, quantile,q.upper)
    }
    
    # Store results
    res$mmean=apply(res$mdraws,2,mean)
    res$msd=apply(res$mdraws,2,sd)
    res$m.5=apply(res$mdraws,2,quantile,0.5)
    res$m.lower=apply(res$mdraws,2,quantile,q.lower)
    res$m.upper=apply(res$mdraws,2,quantile,q.upper)
  
    #Save the list of posterior draws -- each list element is an npost X n dataframe 
    res$wdraws = wt_list 
    
    #Get model mixing results
    res$wmean=mean_matrix
    res$wsd=sd_matrix
    res$w.5=med_matrix
    res$w.lower=lb_matrix
    res$w.upper=ub_matrix
    
  }
  
  if(dosdraws){
    fnames=list.files(fit$folder,pattern=paste(fit$modelname,".sdraws*",sep=""),full.names=TRUE)
    res$sdraws=do.call(cbind,sapply(fnames,data.table::fread))
    
    # Get mixing results: mean prediction and sigma
    res$smean=apply(res$sdraws,2,mean)
    res$ssd=apply(res$sdraws,2,sd)
    res$s.5=apply(res$sdraws,2,median)
    res$s.lower=apply(res$sdraws,2,quantile,q.lower)
    res$s.upper=apply(res$sdraws,2,quantile,q.lower)
    
  }

  res$q.lower=q.lower
  res$q.upper=q.upper
  res$modeltype=fit$modeltype
  
  class(res)="OpenBT_predict"
  
  return(res)
}


#------------------------------------------------
# Get Posterior of calibration parameters
#------------------------------------------------
posterior.openbtmixing = function(fit,q.lower=0.025,q.upper=0.975){
  if(is.null(fit)) stop("No fitted model specified!\n")
  pu = length(unique(fit$uc_col_list))

  # Read in results
  res = list()
  fnames=list.files(fit$folder,pattern=paste(fit$modelname,".udraws*",sep=""),full.names=TRUE)
  
  res$udraws=do.call(cbind,sapply(fnames,data.table::fread))  

  # remove first row, which was the last draw of u from the burn-in (used in prediction file)  
  res$udraws = matrix(res$udraws[-(1:pu),], ncol = pu, byrow = TRUE) 
  
  # Get summary stats
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


#------------------------------------------------
# Tree Scanning Functions
#------------------------------------------------
# Scan the trees in the posterior to extract tree properties
# Returns the mean trees as a list of lists in object mt
# and the variance trees as a list of lists in object st.
# The format is mt[[i]][[j]] is the jth posterior tree from the ith posterior
# sum-of-trees (ensemble) sample.
# The tree is encoded in 4 vectors - the node ids, the node variables,
# the node cutpoints and the node thetas.
scanpost.openbtmixing<-function(post,K){
  fp=file(paste(post$folder,"/",post$modelname,".fit",sep=""),open="r")
  if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$nd) stop("Error scanning posterior\n")
  if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$m) stop("Error scanning posterior\n")
  if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$mh) stop("Error scanning posterior\n")
  if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$nd*post$m) stop("Error scanning posterior\n")
  
  # scan mean trees
  numnodes=scan(fp,what=integer(),nmax=post$nd*post$m,quiet=TRUE)
  lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
  ids=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
  lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
  vars=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
  lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
  cuts=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
  lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
  thetas=scan(fp,what=double(),nmax=lenvec,quiet=TRUE)
  
  # scan var trees
  if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$nd*post$mh) stop("Error scanning posterior\n")
  snumnodes=scan(fp,what=integer(),nmax=post$nd*post$mh,quiet=TRUE)
  lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
  sids=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
  lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
  svars=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
  lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
  scuts=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
  lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
  sthetas=scan(fp,what=double(),nmax=lenvec,quiet=TRUE)
  
  close(fp)
  
  # Now rearrange things into lists of lists so its easier to manipulate
  mt=list()
  ndx=2
  cs.numnodes=c(0,cumsum(numnodes))
  for(i in 1:post$nd) {
    ens=list()
    for(j in 1:post$m)
    {
      tree=list(id=ids[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                var=vars[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                cut=cuts[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                theta=thetas[(K*cs.numnodes[ndx-1]+1):(K*cs.numnodes[ndx])])
      ens[[j]]=tree
      ndx=ndx+1
    }
    mt[[i]]=ens
  }
  
  
  st=list()
  ndx=2
  cs.numnodes=c(0,cumsum(snumnodes))
  for(i in 1:post$nd) {
    ens=list()
    for(j in 1:post$mh)
    {
      tree=list(id=sids[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                var=svars[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                cut=scuts[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                theta=sthetas[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]])
      ens[[j]]=tree
      ndx=ndx+1
    }
    st[[i]]=ens
  }
  
  return(list(mt=mt,st=st))
}


# predict from single tree using scanned result
# scan == mt or st lists
# N = which draw of the MCMC
# j = which tree
singletree.openbtmixing = function(scan,N,j,xvec,xicuts){
  # Define terms
  tree = scan[[N]][[j]]
  cvec = tree$cut
  vvec = tree$var
  K = length(tree$theta)/length(tree$id)
  
  # Get prediction
  h = 1
  id = tree$id[h]
  c0 = cvec[h]
  while(c0>0){
    v0 = vvec[h] + 1
    cp = xicuts[[v0]][c0+1]
    if(xvec[v0]<cp){
      # Left node
      id = tree$id[h+1]
    }else{
      # Right node
      id = tree$id[h+1] + 1
    }
    # Update to new current node
    h = which(tree$id == id)
    c0 = cvec[h]
  }
  
  # Get the theta at position h
  outtheta = tree$theta[(K*(h-1)+1):(K*h)] 
  return(outtheta)
}


# Tree predictions -- get predictions for each individual tree in the sum for an iteration of the MCMC
gettrees.openbtmixing = function(scan,N,j,x,f,xicuts){
  tree_fit = vector("list",fit$m)
  tree_wts = vector("list",fit$m)
  n = nrow(x)
  K = ncol(f)
  wts = matrix(0,nrow=n,ncol=K)
  pred = 0
  for(j in 1:fit$m){
    for(i in 1:n){
      wts[i] = singletree.openbtmixing(scan,ind,j,x[i,],xicuts)
      pred[i] = sum(f[i,]*wts[i,])
    }
    tree_wts[[j]] = wts
    tree_fit[[j]] = pred
  }
  out = list(tree_fit = tree_fit, tree_wts = tree_wts)
  return(out)
}


# Get residuals each tree was regressed on
# tree0 = last fit, tree1 = current fit
treeresid.openbtmixing = function(tree_list0, tree_list1,y){
  # Get residuals
  m = length(tree_list0$tree_fit)
  tree_resid = vector("list",fit$m)
  for(j in 1:m){
    last_ind = (j+1):m
    cur_ind = setdiff(1:m,c(last_ind,j))
    tree_resid[[j]] = y  
    if(last_ind[1] <= m){
      for(l in last_ind){
        tree_resid[[j]] = tree_resid[[j]] - tree_list0$tree_fit[[l]]    
      }
    }
    if(length(cur_ind)>0){
      for(l in cur_ind){
        tree_resid[[j]] = tree_resid[[j]] - tree_list1$tree_fit[[l]]    
      }  
    }
  }
  return(tree_resid)
}


## Find terminal nodes from scanned tree
tnodes_from_scan.openbtmixing = function(tree_list){
  vlist = tree_list$var
  idlist = tree_list$id
  tnodes = c()
  tnodes_theta = matrix(0,nrow = 0, ncol = 2) 
  
  if(length(idlist)>1){
    for(i in 1:(length(idlist)-1)){
      if(idlist[i+1] != 2*idlist[i]){
        tnodes = c(tnodes, idlist[i])
        tnodes_theta = rbind(tnodes_theta,matrix(tree_list$theta[(2*(i-1)+1):(i*2)], 
                                                 nrow = 1, ncol = 2))  
      }
    }
  }
  # Append the last node in the list to the terminal node info
  sz = length(idlist)
  tnodes = c(tnodes,idlist[sz])
  tnodes_theta = rbind(tnodes_theta,matrix(tree_list$theta[(2*(sz-1)+1):(sz*2)], nrow = 1, ncol = 2))
  
  out = list(tnodes = tnodes, tnodes_theta = tnodes_theta)
  return(out)
}


# Get the tree stats -- now limited to number of terminal nodes
get_tree_stats.openbtmixing = function(scan){
  N = length(scan)
  m = length(scan[[1]])
  tnode_count = matrix(0,ncol = m, nrow = N)
  for(i in 1:N){
    tnode_count[i,] = unlist(lapply(scan[[i]],function(x) length(tnodes_from_scan.openbtmixing(x)$tnodes)))
  }
  return(tnode_count)
}


# Get gamma posteriors from rpath model
gammapost.openbtmixing = function(fit){
  if(is.null(fit)) stop("No fitted model specified!\n")
  m = fit$m
  
  # Read in results
  res = list()
  fnames=list.files(fit$folder,pattern=paste(fit$modelname,".rpg*",sep=""),full.names=TRUE)
  
  res$gdraws=do.call(cbind,sapply(fnames,data.table::fread))  
  res$gdraws = matrix(res$gdraws, ncol = m, byrow = TRUE) 
}


# Variogram for random path mixing
variogram.openbtmixing = function(xbnds,hgrid,nd,m,k,base,power,a1,a2,q,gam=NULL,maxd=999,ncut=100,modelname="model"){
  # Data and null values
  if(is.null(gam)){
    const_gamma = FALSE
    gam = a1/(a1+a2) # default value just used as a place holder to start 
  }else{
    const_gamma = TRUE
  }
  
  if(!is.matrix(xbnds)){xbnds = matrix(xbnds, ncol = 2)}
  p = nrow(xbnds)
  
  if(ncol(xbnds)!=2){stop("Error: dimension of xbnds must be px2.")}

  # Compute priors
  tau2 = (1/(2*k*sqrt(m)))^2
  
  #--------------------------------------------------
  # Cut points
  xi=vector("list",p)
  minx_temp=apply(xbnds,1,min)
  maxx_temp=apply(xbnds,1,max)
  
  maxx = round(maxx_temp,1) + ifelse((round(maxx_temp,1)-maxx_temp)>0,0,0.1)
  minx = round(minx_temp,1) - ifelse((minx_temp - round(minx_temp,1))>0,0,0.1)
  for(i in 1:p){
    xinc=(maxx[i]-minx[i])/(ncut+1)
    xi[[i]]=(1:ncut)*xinc+minx[i]
  }
  
  #--------------------------------------------------
  # #write out config file
  xroot="xvg"
  hroot="hvg"
  xiroot="xi"
  folder=tempdir(check=TRUE)
  if(!dir.exists(folder)) dir.create(folder)
  tmpsubfolder=tempfile(tmpdir="")
  tmpsubfolder=substr(tmpsubfolder,5,nchar(tmpsubfolder))
  tmpsubfolder=paste("openbt",tmpsubfolder,sep="")
  folder=paste(folder,"/",tmpsubfolder,sep="")
  if(!dir.exists(folder)) dir.create(folder)
  fout=file(paste(folder,"/config.variogram",sep=""),"w")
  writeLines(c(modelname,paste(nd),paste(m),paste(p),paste(tau2),
               paste(base),paste(power),paste(maxd),paste(a1),paste(a2),paste(q),
               paste(gam),const_gamma,xiroot,xroot,hroot
            ),fout)
  close(fout)
  
  #--------------------------------------------------
  # Write data
  write(t(xbnds),file=paste(folder,"/",xroot,sep=""))
  write(hgrid,file=paste(folder,"/",hroot,sep=""))
  for(i in 1:p) write(xi[[i]],file=paste(folder,"/",xiroot,i,sep=""))
  
  #--------------------------------------------------
  #run variogram program  -- it's not actually parallel so no call to mpirun.
  runlocal=FALSE
  if(Sys.which("openbtcli")[[1]]=="") # not installed in a global locaiton, so assume current directory
    runlocal=TRUE
  
  if(runlocal){
    cmd=paste("./openbtvariogram ",folder,sep="")
  }else{
    cmd=paste("openbtvariogram ",folder,sep="")
  }
  #cmd=paste("./openbtvartivity",sep="")
  system(cmd)
  #system(paste("rm -f ",folder,"/config.variogram",sep=""))
  
  #--------------------------------------------------
  # Read in results
  res=list()
  res$vdraws=do.call(cbind,sapply(paste(folder,"/",modelname,".variogram",sep=""),data.table::fread))
  #res$vdraws=read.table(paste(folder,"/",modelname,".variogram",sep=""))
  res$vdraws = matrix(res$vdraws, ncol = length(hgrid), byrow = TRUE) 
  res$vmean = apply(res$vdraws,2,mean)
  return(res)
}


# R Prediction Function for mixing using the wdraws 
# Required - f_test is in the same order as x_test used in fitw
predict_from_wdraws.openbtmixing=function(fitw, f_test, q.lower = 0.025,q.upper = 0.975){
  # Control
  K = ncol(f_test)
  n_test = nrow(f_test)
  n_draws = nrow(fitw$wdraws[[1]])
  pred_draws = matrix(0, nrow = n_draws, ncol = n_test)

  # Loop through each weight funtion
  for(j in 1:K){
    pred_draws = pred_draws + t(t(fitw$wdraws[[j]])*f_test[,j])
    cat("Weight ",j," done...\n")
  }
  pred_mean = apply(pred_draws,2,mean)
  pred_lb = apply(pred_draws,2,quantile, q.lower)
  pred_ub = apply(pred_draws,2,quantile, q.upper)
  
  out = list(pred_mean = pred_mean, pred_lb = pred_lb, pred_ub = pred_ub)
  return(out)
}

