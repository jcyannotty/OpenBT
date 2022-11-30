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
    mcinfo& mci=static_cast<mcinfo&>(si);
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
void mcbrt::add_observation_to_suff(diterator& diter, sinfo& si)
{
   mcinfo& mci=static_cast<mcinfo&>(si);
   double w;
   w=1.0/(ci.sigma[*diter]*ci.sigma[*diter]);
   mci.n+=1;
   //mci.sumw+=w;
   //mci.sumwy+=w*diter.gety();
}


//--------------------------------------------------
//local_getsuff used for birth.
void mcbrt::local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir)    
{
    double *xx; // current x
    tree::tree_p xbn; // store bottom node of x
    tree::tree_p subtree; // subtree pointer root
    tree::npv subbnv; // bottom nodes for the subtree of interest
    tree::npv uroots; // roots to subtree(s)
    std::vector<mcinfo*> mcv; // mcinfo vector used for subtrees
    sil.n=0; sir.n=0;

    // Cast suff stats to mcinfo -- this usually happens in add_obs_to_suff
    // but we need it here to set 
    mcinfo& mcr=static_cast<mcinfo&>(sir);
    mcinfo& mcl=static_cast<mcinfo&>(sil);

    // Get roots of the subtree(s) in t
    t.getsubtreeroots(uroots, uvec);
    // Check is the node nx in a subtree
    t.nodeinsubtree(nx,uroots,subtree);

    // If nx is not in a subtree - i.e. subtree == 0
    if(!subtree){
        // The left and right node suff stats can be computed as normal 
        for(;diter<diter.until();diter++){
            xx = diter.getxp();
            if(nx==t.bn(diter.getxp(),*xi)){ //does the bottom node = xx's bottom node
                if(xx[v] < (*xi)[v][c]){
                    //sil.n +=1;
                    add_observation_to_suff(diter,sil);
                } else {
                    //sir.n +=1;
                    add_observation_to_suff(diter,sir);
                }
            }
        }
    }else{
        // nx is a node in a subtree, so the birth ratio compares the two subtrees
        // Get bottom nodes in this tree
        subtree->getbots(subbnv);
        
        // initialize suff stat vector 
        std::map<tree::tree_cp,size_t> bnmap;

        // Resize mcv to be the number of term nodes in the subtree which are not changing
        mcv.resize(subbnv.size()-1); 
        size_t j = 0;
        size_t ni = 0; //node index
        for(size_t i=0;i<subbnv.size();i++){ 
            // Is this a different node in the subtree other than nx
            if(subbnv[i] != nx){
                // If so, initialize its sufficient stat instance 
                bnmap[subbnv[i]]=j; 
                mcv[j] = new mcinfo();
                j+=1;
            }  
        }

        for(;diter<diter.until();diter++){
            // Get x and its bottom node pointer
            xx = diter.getxp();
            xbn = t.bn(diter.getxp(),*xi);
            // Is x's term node xbn in the subtree
            if(std::find(subbnv.begin(), subbnv.end(), xbn) != subbnv.end()) { 
                // The node xbn is one of the other ones in the subtree
                if(xbn != nx){
                    ni = bnmap[xbn];
                    add_observation_to_suff(diter,*(mcv[ni]));
                }else{
                    // xbn is the one we are splitting on so handle the left and right
                    if(xx[v] < (*xi)[v][c]) {
                        add_observation_to_suff(diter,mcl);
                    }else{
                        add_observation_to_suff(diter,mcr);
                    }
                }   
            }
        } 
        // Now add the suff stats from the unchanged part of the subtree in sir (could use sil wlog)   
        mcr.setsubtreeinfo(mcv,ci.mu1,ci.tau1);        
    }
}


//--------------------------------------------------
//getsuff used for death
void mcbrt::local_getsuff(diterator& diter, tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir)
{
    tree::tree_p xbn; // bottom node pointer for an x
    tree::tree_p subtree; // subtree pointer root
    tree::npv subbnv; // bottom nodes for the subtree of interest
    tree::npv uroots; // roots to subtree(s)
    std::vector<mcinfo*> mcv; // mcinfo vector used for subtrees
    sil.n=0; sir.n=0;

    // Cast suff stats to mcinfo -- this usually happens in add_obs_to_suff
    // but we need it here to set 
    mcinfo& mcr=static_cast<mcinfo&>(sir);
    mcinfo& mcl=static_cast<mcinfo&>(sil);

    // Get roots of the subtree(s) in t
    t.getsubtreeroots(uroots, uvec);
    
    // Check is the the parent in a subtree
    t.nodeinsubtree(l->p,uroots,subtree);

    // Now check to see if p is the root of the subtree, if so we can just set subtree to 0  
    if(l->p == subtree){subtree == 0;}
    if(!subtree){
        // Do the regular steps
        for(;diter<diter.until();diter++){
            tree::tree_cp bn = t.bn(diter.getxp(),*xi);
            if(bn==l) {
                //sil.n +=1;
                add_observation_to_suff(diter,sil);
            }
            if(bn==r) {
                //sir.n +=1;
                add_observation_to_suff(diter,sir);
            }
        }
    }else{
        // handle the special case for subtree
        // nx is a node in a subtree, so the birth ratio compares the two subtrees
        // Get bottom nodes in this tree
        subtree->getbots(subbnv);
        
        // initialize suff stat vector 
        std::map<tree::tree_cp,size_t> bnmap;

        // Resize mcv to be the number of term nodes in the subtree which are not changing
        mcv.resize(subbnv.size()-2); 
        size_t j = 0;
        size_t ni = 0; //node index
        for(size_t i=0;i<subbnv.size();i++){ 
            // Is this a different node in the subtree other than l or r
            if(subbnv[i] != l && subbnv[i] != r){
                // If so, initialize its sufficient stat instance 
                bnmap[subbnv[i]]=j; 
                mcv[j] = new mcinfo();
                j+=1;
            }  
        }

        for(;diter<diter.until();diter++){
            // Get x and its bottom node pointer
            xbn = t.bn(diter.getxp(),*xi);
            // Is x's term node xbn in the subtree
            if(std::find(subbnv.begin(), subbnv.end(), xbn) != subbnv.end()) { 
                // The node xbn is one of the other ones in the subtree
                if(xbn != l && xbn != r){
                    ni = bnmap[xbn];
                    add_observation_to_suff(diter,*(mcv[ni]));
                }else{
                    // xbn is l or r
                    if(xbn == l) {
                        add_observation_to_suff(diter,mcl);
                    }else{
                        add_observation_to_suff(diter,mcr);
                    }
                }   
            }
        } 
        // Now add the suff stats from the unchanged part of the subtree in sir (could use sil wlog)   
        mcr.setsubtreeinfo(mcv,ci.mu1,ci.tau1);
    }
    
}


//--------------------------------------------------
//local_subsuff
void mcbrt::local_subsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv)
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