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
