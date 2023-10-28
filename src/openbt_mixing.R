#------------------------------------------------
# Openbt Mixing R interface
#------------------------------------------------
# Get Posterior of calibration parameters
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
