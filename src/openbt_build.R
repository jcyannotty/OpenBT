#------------------------------------------------
# OpenBT Build from R
# This is a protype for building openbt locally using R
# Current design is for the OSU Unity cluster and also assumes 
# Eigen is included in the openbt/src directory.
#------------------------------------------------
setwd("home/yannotty.1/openbt/src")
system("make")
