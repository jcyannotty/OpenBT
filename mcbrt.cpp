// mcbrt.cpp: model class for mean calibration
#include <iostream>
#include <map>

//header files from OpenBT 
#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"
#include "mcbrt.h"

//Include the Eigen header files
#include "Eigen/Dense"
#include <Eigen/StdVector>

//--------------------------------------------------
//a single iteration of the MCMC for brt model
void mcbrt::drawvec(rn& gen){
    //cout << "enter drawvec" << endl; 
    //All the usual steps
    brt::drawvec(gen);

    // Update the in-sample predicted vector
    setf_vec();

    // Update the in-sample residual vector
    setr_vec();
}

//--------------------------------------------------
//slave controller for draw when using MPI
void mcbrt::drawvec_mpislave(rn& gen){
    //All the usual steps
    brt::drawvec_mpislave(gen);

    // Update the in-sample predicted vector
    setf_vec();

    // Update the in-sample residual vector
    setr_vec();
}

//--------------------------------------------------
//draw theta for a single bottom node for the brt model
vxd mcbrt::drawnodethetavec(sinfo& si, rn& gen){
    //initialize variables
    mcsinfo& mcsi=static_cast<mcsinfo&>(si);
    mxd I(k,k), Sig_inv(k,k), Sig(k,k), Ev(k,k), E(k,k), Sp(k,k), Prior_Sig_Inv(k,k);
    vxd muhat(k), evals(k), stdnorm(k), betavec(k);
    double tau1_sqr = ci.tau1*ci.tau1;
    double tau2_sqr = ci.tau2*ci.tau2;
    
    I = Eigen::MatrixXd::Identity(k,k); //Set identity matrix
    //betavec = ci.mu1*Eigen::VectorXd::Ones(k); //Set the prior mean vector

    //Print out matrix algebra step-by-step
    /*
    std::cout << "\nAll matrix Calculations:" << std::endl;
    std::cout << "Sig_inv = \n" << Sig_inv << std::endl;
    std::cout << "\n Sig = \n" << Sig << std::endl;
    std::cout << "\n muhat = \n" << muhat << std::endl;
    std::cout << "\n Ev = \n" << Ev << std::endl;
    std::cout << "\n evals = \n" << evals << std::endl;
    std::cout << "\n E = \n" << E << std::endl;
    std::cout << "\n Sp = \n" << Sp << std::endl;
    std::cout << "\n thetavec = " << std::endl;
    */
        
    //--Generate the MVN vector
    //return muhat + Sp*stdnorm;
}

//--------------------------------------------------
//local_getsuff used for birth.
void brt::local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir)    
{
    double *xx;//current x
    sil.n=0; sir.n=0;

    // Check is the node nx in a subtree? void checksubtree(...)

    for(;diter<diter.until();diter++)
    {
        xx = diter.getxp();
        if(nx==t.bn(diter.getxp(),*xi)) { //does the bottom node = xx's bottom node
            if(xx[v] < (*xi)[v][c]) {
                //sil.n +=1;
                add_observation_to_suff(diter,sil);
            } else {
                //sir.n +=1;
                add_observation_to_suff(diter,sir);
            }
        }
    }
}

//--------------------------------------------------
//local_subsuff
void brt::local_subsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv)
{
   tree::tree_cp tbn; //the pointer to the bottom node for the current observation
   size_t ni;         //the  index into vector of the current bottom node
   size_t index;      //the index into the path vector.
   double *x;
   tree::tree_p root=path[path.size()-1];

   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   siv.clear();
   siv.resize(nb);

   std::map<tree::tree_cp,size_t> bnmap;
   std::map<tree::tree_cp,size_t> utreemap;
   for(bvsz i=0;i!=bnv.size();i++) { bnmap[bnv[i]]=i; siv[i]=newsinfo(); }

   for(;diter<diter.until();diter++) {
      index=path.size()-1;
      x=diter.getxp();
      if(root->xonpath(path,index,x,*xi)) { //x is on the subtree, 
         tbn = nx->bn(x,*xi);              //so get the right bn below interior node n.
         ni = bnmap[tbn];
         //siv[ni].n +=1;
         add_observation_to_suff(diter, *(siv[ni]));
      }
      //else this x doesn't map to the subtree so it's not added into suff stats.
   }
}