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
