// mcbrt.cpp: model class for mean calibration
#include <iostream>
#include <map>
#include <cmath>

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
void mcbrt::add_observation_to_suff(diterator& diter, sinfo& si)
{
    mcinfo& mci=static_cast<mcinfo&>(si); // May or may not need this -- casting occurs outside of this function for calibration
    double w;
    
    // Get the precision and the increment the sample sizes
    w=1.0/(ci.sigma[*diter]*ci.sigma[*diter]);
    mci.n+=1-(*fi)(*diter,1); // This adds 0 if field obs and 1 if model run
    mci.nf+=(*fi)(*diter,1); // This adds 1 if field obs and 0 if model run
   
    // If this is a field observation the fi == 1, else it is a model run
    if((*fi)(*diter,1) == 1){
        mci.sumyw+=w*diter.gety();
        mci.sumwf+=w;
        //cout << "sumwf = " << mci.sumwf << endl;
    }else{
        mci.sumzw+=w*diter.gety();
        mci.sumwc+=w;
        //cout << "sumwc = " << mci.sumwc << endl;
    }
    //if(mci.subtree_info) cout << "subtree? " << mci.subtree_info << "---" << rank << endl;
}

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
void mcbrt::drawthetavec(rn& gen)
{
    tree::npv bnv;
    std::vector<sinfo*>& siv = newsinfovec();
    tree::npv uroots; // roots to subtree(s)
    tree::tree_p subtree;
    std::map<tree::tree_cp,std::vector<size_t>> unmap; // subtree map
    std::vector<size_t> xnodes; // non-subtree nodes
    std::vector<sinfo*> unodestats; // subtree suff stats  
    size_t ind;
    std::vector<std::vector<double>> orthogarea;
    std::vector<double> theta1_modvec;

    // Initialize theta1 and theta2 for the subtrees
    double theta1, theta2;
    vxd thetavec(2);
    // Get all suff stats per node
    allsuff(bnv,siv);

    // Find the subtrees -- Get roots of the subtree(s) in t
    t.getsubtreeroots(uroots, uvec);
    
    // Check is the node in bnv is in a subtree
    for(size_t i=0;i<bnv.size();i++){
        subtree = 0;
        bnv[i]->nodeinsubtree(uroots,subtree);
        if(subtree){
            // This node is in a subtree, append i to the corresponding map
            unmap[subtree].push_back(i);
        }else{
            // Not in a subtree
            xnodes.push_back(i);
        }
    }

    // Get the scales if using orthogonal discrepancy
    if(orth_delta){
        orthogarea = get_orthogonal_scales_draw(bnv,uroots,unmap,xnodes);
        for(size_t i=0;i<bnv.size();i++){
            mcinfo& mci=static_cast<mcinfo&>(*(siv[i]));
            mci.setdeltaarea(orthogarea[1][i],orthogarea[0]);
        }
    }

#ifdef _OPENMPI
    mpi_resetrn(gen);
#endif
    // Draw theta1 and theta2 from non subtree nodes
    for(size_t i=0;i<xnodes.size();i++) {
        // add modularization condition here....
        if(modular){
            bnv[xnodes[i]]->setthetavec(drawnodethetavec_modular(*(siv[xnodes[i]]),gen));
        }else{
            bnv[xnodes[i]]->setthetavec(drawnodethetavec(*(siv[xnodes[i]]),gen));
        }
    }

    // add second modularization step for subtree here....
    // Loop through subtrees (the ids stored in uroots) and draw thetas
    // Get all bot nodes in the subtree
    if(modular){
        // Modularization 
        for(size_t i=0;i<uroots.size();i++){
            // Get all bot nodes in the subtree
            unodestats.clear();
            theta1_modvec.clear();
            for(size_t j=0;j<unmap[uroots[i]].size();j++){
                ind = unmap[uroots[i]][j]; // bottom node/suff stat index
                unodestats.push_back(siv[ind]);
            }
            // Draw theta1 for the each node in subtree
            for(size_t j=0;j<unodestats.size();j++){
                ind = unmap[uroots[i]][j]; // bottom node/suff stat index
                theta1 = drawtheta1_modular(*(siv[ind]),gen);
                theta1_modvec.push_back(theta1);
            }
            // Draw theta2
            theta2 = drawtheta2_modular(unodestats,theta1_modvec,gen);
            // Set thetavec with individual theta1 and common theta2
            for(size_t j=0;j<unodestats.size();j++){
                ind = unmap[uroots[i]][j];
                thetavec(0) = theta1_modvec[j];
                thetavec(1) =  theta2;
                bnv[ind]->setthetavec(thetavec);
            }
        }
    }else{
        // Regular Sampler
        for(size_t i=0;i<uroots.size();i++){
            // Get all bot nodes in the subtree
            unodestats.clear();
            for(size_t j=0;j<unmap[uroots[i]].size();j++){
                ind = unmap[uroots[i]][j]; // bottom node/suff stat index
                unodestats.push_back(siv[ind]);
            } 
            // Draw theta2 for the subtrees
            theta2 = drawtheta2(unodestats,gen);
            
            // Draw theta1 per node conditional on theta2
            for(size_t j=0;j<unodestats.size();j++){
                ind = unmap[uroots[i]][j]; // bottom node/suff stat index
                theta1 = drawtheta1(*(siv[ind]),gen,theta2);
                thetavec << theta1, theta2; // create the eigen thetavec
                bnv[ind]->setthetavec(thetavec); // Set thetavec
            }
        }
    }

    delete &siv;  //and then delete the vector of pointers.
}

//--------------------------------------------------
//draw theta for a single bottom node for the brt model
vxd mcbrt::drawnodethetavec(sinfo& si, rn& gen){
    //initialize variables
    mcinfo& mci=static_cast<mcinfo&>(si); // Casting might not be needed since its done earlier
    mxd I(2,2), PostSig(2,2), A(2,2), Ev(2,2), E(2,2), Sp(2,2), Prior_Sig_Inv(2,2), HWinvH(2,2);
    vxd postmean(2), evals(2), stdnorm(2), muvec(2), HWinvR(2);
    double tau1_sqr = ci.tau1*ci.tau1;
    double tau2_sqr = ci.tau2*ci.tau2;
    double suma2 = 0.0;

    // Rescale tau2 if orthogonal delta
    if(orth_delta){
        for(size_t j=0;j<mci.delta_alist.size();j++){suma2+=mci.delta_alist[j]*mci.delta_alist[j];} 
        if(mci.delta_alist.size()>1){
            tau2_sqr = tau2_sqr*(1-mci.delta_area*mci.delta_area/suma2);
        }else{
            tau2_sqr = tau2_sqr*0.05; // minimum variance
        }    
    }
     
    I = Eigen::MatrixXd::Identity(2,2); //Set identity matrix
    
    // Set Prior Precision
    Prior_Sig_Inv = mxd::Zero(2,2);
    Prior_Sig_Inv(0,0) = 1/tau1_sqr;
    Prior_Sig_Inv(1,1) = 1/tau2_sqr;
    muvec << ci.mu1, ci.mu2;
    
    // Set sufficient stats
    //HWinvH = mci.sumwf*I;
    HWinvH = mci.sumwf*mxd::Ones(2,2);
    HWinvH(0,0) = HWinvH(0,0) + mci.sumwc;
    HWinvR << (mci.sumyw + mci.sumzw), mci.sumyw;

    // Get posterior mean and covariance
    A = HWinvH + Prior_Sig_Inv;
    PostSig = A.llt().solve(I);
    postmean = PostSig*(HWinvR + Prior_Sig_Inv*muvec);

    //Spectral Decomposition of Covaraince Matrix Sig -- maybe move to a helper function
    //--Get eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(PostSig); //Get eigenvectors and eigenvalues
    if(eigensolver.info() != Eigen::ComputationInfo::Success) abort(); //Checks if any errors occurred
    evals = eigensolver.eigenvalues(); //Get vector of eigenvalues
    Ev = eigensolver.eigenvectors(); //Get matrix of eigenvectors

    //--Get sqrt of eigen values and store into a matrix    
    E = mxd::Zero(2,2); //Set as matrix of 0's
    E.diagonal() = evals.array().sqrt(); //Diagonal Matrix of sqrt of eigen values
    
    //--Compute Spectral Decomposition
    Sp = Ev*E*Ev.transpose();

    //Generate MVN Random vector
    //--Get vector of standard normal normal rv's
    for(size_t i=0; i<2;i++){
        stdnorm(i) = gen.normal(); 
    }

    //Print out matrix algebra step-by-step
    /*
    std::cout << "\nAll matrix Calculations:" << std::endl;
    std::cout << "Prior_Sig_inv = \n" << Prior_Sig_Inv << std::endl;
    std::cout << "\n PostSig = \n" << PostSig << std::endl;
    std::cout << "\n postmean = \n" << postmean << std::endl;
    std::cout << "\n Ev = \n" << Ev << std::endl;
    std::cout << "\n evals = \n" << evals << std::endl;
    std::cout << "\n E = \n" << E << std::endl;
    std::cout << "\n Sp = \n" << Sp << std::endl;
    */
    //--Generate the MVN vector
    return postmean + Sp*stdnorm;
}

//--------------------------------------------------
// Draw theta2 for a subtree
double mcbrt::drawtheta2(std::vector<sinfo*> sivec, rn &gen)
{
    
    double sumw=0.0, summeanw=0.0, postvar, postmean, w, theta2;
    double tau2_sqr = ci.tau2*ci.tau2;
    double suma2=0.0;
    std::vector<double> mv;

    // Rescale tau2 if orthogonal delta -- same areas for all nodes in subtree
    if(orth_delta){
        mcinfo& mci=static_cast<mcinfo&>(*sivec[0]);
        for(size_t j=0;j<mci.delta_alist.size();j++){suma2+=mci.delta_alist[j]*mci.delta_alist[j];} 
        if(mci.delta_alist.size()>1){
            tau2_sqr = tau2_sqr*(1-mci.delta_area*mci.delta_area/suma2);
        }else{
            tau2_sqr = tau2_sqr*0.05; // minimum variance
        }
    }

    // Get modularized prior information 
    for(size_t i=0;i<sivec.size();i++){
        // cast to mcinfo and get modularized mean and var
        mcinfo& mci=static_cast<mcinfo&>(*sivec[i]); 
        mv = mci.getmoments(ci.mu1, ci.tau1);
        // Compile modularized stats
        if(mv[1]>0){w = 1/mv[1];}else{w = 0.0;}
        //cout << "var = " << mv[1] << endl;
        sumw += w;
        summeanw += mv[0]*w;
    }
    //if(rank==0) cout << "summeanw = " << summeanw << endl;
    // Now get the posterior mean and variance
    postvar = 1/(sumw + 1/tau2_sqr);
    postmean = postvar*(summeanw + ci.mu2/tau2_sqr);
    //if(rank==0) cout << "postmean = " << postmean << endl;
    //if(rank==0) cout << "postvar = " << postvar << endl;
    // Draw a random variable
    theta2 = postmean + gen.normal()*sqrt(postvar); 
    return theta2;
}

//--------------------------------------------------
// Draw theta1 conditional on theta2
double mcbrt::drawtheta1(sinfo &si, rn &gen, double theta2)
{
    mcinfo& mci=static_cast<mcinfo&>(si);
    double mtilde, vtilde;
    double postmean, postvar, theta1;
    double t1_sqr = ci.tau1*ci.tau1;
    
    // Compute vtilde and mtilde
    vtilde = 1/(mci.sumwc + 1/t1_sqr);
    mtilde = vtilde*(mci.sumzw + ci.mu1/t1_sqr);

    // Does this node have field data? (mci.nf>0)
    if(mci.sumwf>0){
        postvar = 1/(mci.sumwf + 1/vtilde);
        postmean = postvar*(mci.sumyw - theta2*mci.sumwf + mtilde/vtilde);
        theta1 = postmean + gen.normal()*sqrt(postvar);
    }else{
        // Only model runs on this node
        theta1 = mtilde + gen.normal()*sqrt(vtilde);
    }
    return theta1;
}


//--------------------------------------------------
// Draw node thetavec modularization style
vxd mcbrt::drawnodethetavec_modular(sinfo& si, rn& gen){
    mcinfo& mci=static_cast<mcinfo&>(si); // Casting might not be needed since its done earlier
    vxd outtheta(2);
    double postmean2 = 0.0, postvar2 = 0.0;
    double theta1 = 0.0, theta2 = 0.0;
    double tau1_sqr = ci.tau1*ci.tau1;
    double tau2_sqr = ci.tau2*ci.tau2;
    double suma2 = 0.0;

    // Get theta1 using the model runs
    theta1 = drawtheta1_modular(mci, gen);

    // Get the theta2 using the field obs
    postvar2 = 1/(mci.sumwf + 1/tau2_sqr);
    postmean2 = postvar2*(mci.sumyw - theta1*mci.sumwf + ci.mu2/tau2_sqr);
    theta2 = postmean2 + sqrt(postvar2)*gen.normal();

    // return the theta1 and theta2 vector
    outtheta << theta1, theta2;
    return outtheta;
}


//--------------------------------------------------
// Modularization: drawtheta1
double mcbrt::drawtheta1_modular(sinfo& si, rn& gen){
    mcinfo& mci=static_cast<mcinfo&>(si);
    double mtilde, vtilde, theta1;
    double t1_sqr = ci.tau1*ci.tau1;
    
    // Compute vtilde and mtilde
    vtilde = 1/(mci.sumwc + 1/t1_sqr);
    mtilde = vtilde*(mci.sumzw + ci.mu1/t1_sqr);

    // Only model runs on this node
    theta1 = mtilde + gen.normal()*sqrt(vtilde);
    //cout << "mtilde = " << mtilde << endl;
    //cout << "vtilde = " << vtilde << endl;
    return theta1;
}


//--------------------------------------------------
// Modularization: drawtheta2
double mcbrt::drawtheta2_modular(std::vector<sinfo*> sivec, std::vector<double> &theta1_vec, rn& gen){
    double sumw=0.0, summeanw=0.0, postvar, postmean, w, theta2, theta1;
    double rhat, vhat;
    double tau2_sqr = ci.tau2*ci.tau2;
    double suma2=0.0;

    // Get modularized prior information 
    for(size_t i=0;i<sivec.size();i++){
        // cast to mcinfo and get modularized mean and var
        mcinfo& mci=static_cast<mcinfo&>(*sivec[i]); 
        // Compile suff stats from field obs
        if(mci.nf>0){theta1 = theta1_vec[i];}else{theta1 = 0.0;} // swtich back to mci.sumwf > 0 condition
        sumw += mci.sumwf;
        summeanw += mci.sumyw - theta1*mci.sumwf;
        /*
        if(mci.sumwf>0){
            //rhat = mci.sumyw/mci.sumwf;
            //vhat = 1/mci.sumwf;
            w = 1/vhat;
        }else{
            w = 0.0;
        }        
        
        // Store the collected suff stats
        sumw += w;
        summeanw += (rhat - theta1)*w;
        */
    }
    // compute rhat and vhat 
    //vhat = 1/sumw;
    //rhat = summeanw*vhat;
    
    // Now get the posterior mean and variance
    postvar = 1/(sumw + 1/tau2_sqr);
    postmean = postvar*(summeanw + ci.mu2/tau2_sqr);

    // Draw a random variable
    theta2 = postmean + gen.normal()*sqrt(postvar); 
    return theta2;
}

    
//--------------------------------------------------
// Set f for mcbrt, also sets eta
void mcbrt::local_setf_vec(diterator& diter)
{
    tree::tree_p bn;
    vxd thetavec_temp(k); //Initialize a temp vector to facilitate the fitting
    etahat.resize((*di).n,0); //reset eta
    for(;diter<diter.until();diter++) {
        bn = t.bn(diter.getxp(),*xi);
        thetavec_temp = bn->getthetavec(); 
        yhat[*diter] = (*fi).row(*diter)*thetavec_temp;
        etahat[*diter] = thetavec_temp(0); // eta is the first element      
    }
}

//--------------------------------------------------
// Compute log marginal likelihood for calibration model
// Outside of subtrees, this is straitforward and relative to the selected node
// With subtrees - the lm has three potential parts:
// (1) lm from model runs
// (2) lm from field obs, consdering all relevant terminal nodes in the subtrees (those where nf>0) 
// (3) The sufficient statistic kernel g(rf)
double mcbrt::lm(sinfo& si)
{
    // Initialize terms for lm
    mcinfo& mci=static_cast<mcinfo&>(si); // cast si into mcinfo
    double lmstree = 0.0, lmn = 0.0;
    
    // Get subtree lm if node contains subtree info        
    if(mci.subtree_info){
        lmstree = lmsubtree(mci);
        //cout << "subtree lm " << lmn << "-----" << rank << endl;
    }
    // Add individual node contribution -- checking subtree condition
    if(mci.subtree_node){
        lmn = lmsubtreenode(mci);
        //cout << "subtree node lmn1 " << lmn << "-----" << rank << endl;
    }else{
        lmn = lmnode(mci); // lm for nodes outside of subtree, used when no subtree is involved
        //cout << "subtree node lmn2 " << lmn << "-----" << rank << endl;
    }
    return lmstree + lmn;
}

//--------------------------------------------------
// Integrated likelihood for an individual node outside of a subtree
double mcbrt::lmnode(mcinfo &mci){
    // Initialize terms
    double tau1_sqr = ci.tau1*ci.tau1;
    double tau2_sqr = ci.tau2*ci.tau2;
    double siginv_det, q, v; 
    double suma2=0.0;
    mxd Sig(2,2);
    vxd meanvec(2);

    // Initialize lm values
    double lmout = 0.0;

    // Rescale tau2 if orthogonal delta
    if(orth_delta){
        for(size_t j=0;j<mci.delta_alist.size();j++){suma2+=mci.delta_alist[j]*mci.delta_alist[j];} 
        if(mci.delta_alist.size()>1){
            tau2_sqr = tau2_sqr*(1-mci.delta_area*mci.delta_area/suma2);
        }else{
            tau2_sqr = tau2_sqr*0.05; // minimum variance
        }
    }

    // Does this node have field observations 
    if(mci.sumwf>0){
        // Get determinant of the inverse posterior variance (have this in closed form bc 2x2)
        siginv_det = (mci.sumwf + mci.sumwc + 1/tau1_sqr)*(mci.sumwf + 1/tau2_sqr) - mci.sumwf*mci.sumwf;
        
        // Get the Sigma matrix ((Ht*Winv*H) + priorsig^inv)^inv
        Sig(0,0) = mci.sumwf + 1/tau2_sqr;
        Sig(1,0) = -mci.sumwf;
        Sig(0,1) = -mci.sumwf; 
        Sig(1,1) = mci.sumwf + mci.sumwc + 1/tau1_sqr;
        Sig = Sig/siginv_det;

        // Get the mean vector
        meanvec(0) = mci.sumyw + mci.sumzw + ci.mu1/tau1_sqr;
        meanvec(1) = mci.sumyw + ci.mu2/tau2_sqr;

        // Get the quadratic term
        q = meanvec.transpose()*Sig*meanvec;

        // This node has model runs and field obs
        lmout = -0.5*(log(tau1_sqr) + log(tau2_sqr) + log(siginv_det)); // variance terms
        lmout += -0.5*(ci.mu1*ci.mu1/tau1_sqr + ci.mu2*ci.mu2/tau2_sqr - q); // mean terms (ignoring the part that cancels in a ratio)
    }else{
        // This node just has model runs (ignoring terms which cancel in the ratio)
        v = tau1_sqr*mci.sumwc + 1;
        lmout = -0.5*(ci.mu1*ci.mu1/tau1_sqr); // prior mean term
        lmout += 0.5*tau1_sqr*(mci.sumzw + ci.mu1*ci.mu1/tau1_sqr)*(mci.sumzw + ci.mu1*ci.mu1/tau1_sqr)/v; // mean term
        lmout += -0.5*log(v); //variance term
    }

    return lmout;
}

//--------------------------------------------------
// integrated likelihood for a mcinfo instance which is storing the joint information across the nodes
// p(rhat1,...,rhatB | Tj, u, sigma2)
double mcbrt::lmsubtree(mcinfo &mci){
    // Initialize terms for lm
    double tau2_sqr = ci.tau2*ci.tau2;

    // Initialize terms for total lm from each piece
    double lmstree = 0.0;

    // Initialize terms use to compile sufficient stats
    std::vector<double> nodemv;
    double sum_meansqrw = 0.0;
    double sum_meanw = 0.0;
    double sum_w = 0.0;
    double sum_logw = 0.0;
    double w = 1.0;
    double B = 0; // number of nodes in the subtree
    double suma2=0.0;

    // Set subtree and sibling moments
    mci.setsubtreemoments(ci.mu1, ci.tau1);

    // Rescale tau2 if orthogonal delta
    if(orth_delta){
        for(size_t j=0;j<mci.delta_alist.size();j++){suma2+=mci.delta_alist[j]*mci.delta_alist[j];} 
        if(mci.delta_alist.size()>1){
            tau2_sqr = tau2_sqr*(1-mci.delta_area*mci.delta_area/suma2);
        }else{
            tau2_sqr = tau2_sqr*0.05; // minimum variance
        }
    }

    // Sum over the vector componenets with non-zero variances (meaning field data is on the ith node)
    for(size_t i = 0;i < mci.subtree_mean.size();i++){
        if(mci.subtree_var[i]>0){
            w = 1/mci.subtree_var[i];
            sum_meansqrw += mci.subtree_mean[i]*mci.subtree_mean[i]*w;
            sum_meanw += mci.subtree_mean[i]*w;
            sum_w+=w;
            sum_logw+=log(w);
            B+=1;
        }
    }
    // Add sibling info if this suff stat instance has it & if it has field obs on it
    if(mci.sibling_info){
        if(mci.sibling_var>0){
            w = 1/mci.sibling_var;
            sum_meansqrw += mci.sibling_mean*mci.sibling_mean*w;
            sum_meanw += mci.sibling_mean*w;
            sum_w+=w;
            sum_logw+=log(w);
            B+=1;
        }
    }
    // Now calculate and add in the mean and var from this node if it has field obs on it
    nodemv = mci.getmoments(ci.mu1, ci.tau1);
    if(nodemv[1]>0){
        w = 1/nodemv[1];
        sum_meansqrw += nodemv[0]*nodemv[0]*w;
        sum_meanw += nodemv[0]*w;
        sum_w+=w;
        sum_logw+=log(w);
        B+=1;
    }

    // Now compute the lm contribution from all nodes in subtree
    lmstree = -0.5*(B*log(2*M_PI) - sum_logw + log(tau2_sqr*sum_w +1)); // Variance terms  **changed second term to - instead of +    
    lmstree += -0.5*(ci.mu2*ci.mu2/tau2_sqr + sum_meansqrw); // mean terms
    lmstree += 0.5*tau2_sqr*(sum_meanw + ci.mu2/tau2_sqr)*(sum_meanw + ci.mu2/tau2_sqr)/(tau2_sqr*sum_w + 1);
    
    return lmstree;
}

//--------------------------------------------------
// integrated likelihood for an individual node in the subtree -- adds its additional information
// g(rhatb) & p(rvec^c) b=1,...,B
double mcbrt::lmsubtreenode(mcinfo &mci){
    // Initialize terms for lm
    double tau1_sqr = ci.tau1*ci.tau1;
    double rhat, vhat;

    // Initialize terms for total lm from each piece
    double lmc = 0.0, lmf =0.0;
    // Contribution from field data (mci.nf>0) 
    if(mci.sumwf>0){
        rhat = mci.sumyw/mci.sumwf;
        vhat = 1/mci.sumwf;
        lmf = 0.5*(log(2*M_PI*vhat) + rhat*rhat/vhat);
    }
    // Model runs mci.n>0 (Previously, mci.n > mci.nf)
    if(mci.sumwc>0){
        lmc = 0.5*tau1_sqr*(mci.sumzw + ci.mu1/tau1_sqr)*(mci.sumzw + ci.mu1/tau1_sqr)/(tau1_sqr*mci.sumwc + 1); //mean term
        lmc += -0.5*ci.mu1*ci.mu1/tau1_sqr; //prior mean term
        lmc += -0.5*log(tau1_sqr*mci.sumwc + 1); //variance term
    }
    return lmc + lmf;
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
    tree::npv obnv; // bottom node vector for orthogonal discrepancy
    std::vector<mcinfo*> mcv; // mcinfo vector used for subtrees
    std::vector<double> area; // area vector for orthogonal discrepancy
    sil.n=0; sir.n=0;
    //cout << "BIRTH" << endl;
    // Cast suff stats to mcinfo -- this usually happens in add_obs_to_suff
    // but we need it here to set 
    mcinfo& mcr=static_cast<mcinfo&>(sir);
    mcinfo& mcl=static_cast<mcinfo&>(sil);

    // Get roots of the subtree(s) in t
    t.getsubtreeroots(uroots, uvec);
    // Check is the node nx in a subtree
    nx->nodeinsubtree(uroots,subtree);
    
    // Now check to see if p is the root of the subtree, if so we can just set subtree to 0. 
    // When nx starts a subtree, we only need to consider the l & r nodes, hence no need to pool information across the rest of the subtree  
    if(nx == subtree){subtree = 0;}
    // If nx is not in a subtree - i.e. subtree == 0 meaning we have a null pointer
    if(!subtree){
        //cout << "mcbrt local getsuff for nonsubtree..." << endl;
        // The left and right node suff stats can be computed as normal 
        // brt::local_getsuff(diter,nx,v,c,sil,sir);
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

        // Set subtree_node = true for left and right children
        mcl.subtree_node = true;
        mcr.subtree_node = true;

        // Resize mcv to be the number of term nodes in the subtree which are not changing
        mcv.resize(subbnv.size()-1);
        //cout << "mcv.size() = " << mcv.size() << endl; 
        size_t j = 0;
        size_t ni = 0; //node index
        for(size_t i=0;i<subbnv.size();i++){ 
            // Is this a different node in the subtree other than nx
            if(subbnv[i] != nx){
                // If so, initialize its sufficient stat instance 
                bnmap[subbnv[i]]=j; 
                mcv[j] = new mcinfo(true); // constructor with true input means this node is in a subtree
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
        mcr.setsubtreeinfo(mcv); 
        // Add the sibling information from mcl to mcr
        mcr.setsiblinginfo(mcl);       
    }

    // Get areas if using orthogonal delta
    if(orth_delta){
        std::vector<double> a(uvec[0],0);
        std::vector<double> b(uvec[0],0);
        std::vector<double> area_bd;
        double areanx = 1.0;

        // Get all bottom nodes
        // Get the area across the rest of the discrepancy partitions excluding nx
        // Then redo the subtree calculation
        t.getbots(obnv);
        area = get_orthogonal_scales_bd(obnv, uroots, nx);
        nx->nodeinsubtree(uroots,subtree);

        // Get the rectangles with respect to x-space for nx (assume u predictors come after all x predictors)
        for(size_t i=0;i<uvec[0];i++){
            // reset upper and lower bounds
            int L,U;
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();

            // Get the interval [L,U] indexes
            if(subtree){
                subtree->rgi(i,&L,&U);
            }else{
                nx->rgi(i,&L,&U);
            }
            
            // Now we have the interval endpoints, put corresponding values in a,b matrices.
            if(L!=std::numeric_limits<int>::min()){ 
                a[i]=(*xi)[i][L];
            }else{
                a[i]=(*xi)[i][0];
            }
            if(U!=std::numeric_limits<int>::max()) {
                b[i]=(*xi)[i][U];
            }else{
                b[i]=(*xi)[i][(*xi)[i].size()-1];
            }
            // Update the area
            areanx = areanx*(b[i] - a[i]); 
        }

        if(subtree){
            area.push_back(areanx);
            mcl.setdeltaarea(areanx, area); // left has same area as parent
            mcr.setdeltaarea(areanx, area); // right has same area as parent
        }else{
            double areal =  areanx;
            double arear =  areanx;
            int L,U;
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
            nx->rgi(v,&L,&U);
            
            // Redefine the left and right areas 
            if(L==std::numeric_limits<int>::min()){L = (*xi)[v][0];}
            if(U==std::numeric_limits<int>::max()){U = (*xi)[v][(*xi)[v].size()-1];}
            
            areal = areal*((*xi)[v][c]-(*xi)[v][L])/((*xi)[v][U]-(*xi)[v][L]); 
            arear = arear*((*xi)[v][U]-(*xi)[v][c])/((*xi)[v][U]-(*xi)[v][L]);
            
            // Add to area vector
            area_bd = area;
            area.push_back(areal); area.push_back(arear);
            area_bd.push_back(areanx);
            // Add areas to suff stats
            mcl.setdeltaarea(areanx, area, area_bd); // left is different than parent
            mcr.setdeltaarea(areanx, area, area_bd); // right is different than parent
        }
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
    //cout << "DEATH" << endl;
    // Cast suff stats to mcinfo -- this usually happens in add_obs_to_suff
    // but we need it here to set 
    mcinfo& mcr=static_cast<mcinfo&>(sir);
    mcinfo& mcl=static_cast<mcinfo&>(sil);

    // Get roots of the subtree(s) in t
    t.getsubtreeroots(uroots, uvec);
    
    // Check is the the parent in a subtree
    (l->p)->nodeinsubtree(uroots,subtree);

    // Now check to see if p is the root of the subtree, if so we can just set subtree to 0. 
    // When p starts a subtree, we only need to consider the l & r nodes, hence no need to pool information across the rest of the subtree  
    if(l->p == subtree){subtree = 0;} // Set to null pointer
    if(!subtree){
        // Do the regular steps
        // brt::local_getsuff(diter,l,r,sil,sir);
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

        // Set subtree_node = true for left and right children
        mcl.subtree_node = true;
        mcr.subtree_node = true;
        
        // Resize mcv to be the number of term nodes in the subtree which are not changing
        mcv.resize(subbnv.size()-2); 
        size_t j = 0;
        size_t ni = 0; //node index
        for(size_t i=0;i<subbnv.size();i++){ 
            // Is this a different node in the subtree other than l or r
            if(subbnv[i] != l && subbnv[i] != r){
                // If so, initialize its sufficient stat instance 
                bnmap[subbnv[i]]=j; 
                mcv[j] = new mcinfo(true);
                j+=1;
            }  
        }
        // Loop through data and assign to relevant bns
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
        mcr.setsubtreeinfo(mcv);
       
        // Add the sibling information from mcl to mcr
        mcr.setsiblinginfo(mcl);
    }

    // Orthogonal discrepancy for death -- CLEAN UP into one or multiple functions!!
    if(orth_delta){
        std::vector<double> a(uvec[0],0);
        std::vector<double> b(uvec[0],0);
        std::vector<double> area_bd;
        double areanx = 1.0;
        std::vector<double> area;
        tree::npv obnv;

        // Get all bottom nodes
        // Get the area across the rest of the discrepancy partitions excluding nx
        // Then redo the subtree calculation
        t.getbots(obnv);
        area = get_orthogonal_scales_bd(obnv, uroots, l->p);
        (l->p)->nodeinsubtree(uroots,subtree);

        // Get the rectangles with respect to x-space for nx (assume u predictors come after all x predictors)
        for(size_t i=0;i<uvec[0];i++){
            // reset upper and lower bounds
            int L,U;
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();

            // Get the interval [L,U] indexes
            if(subtree){
                subtree->rgi(i,&L,&U);
            }else{
                (l->p)->rgi(i,&L,&U);
            }
            
            // Now we have the interval endpoints, put corresponding values in a,b matrices.
            if(L!=std::numeric_limits<int>::min()){ 
                a[i]=(*xi)[i][L];
            }else{
                a[i]=(*xi)[i][0];
            }
            if(U!=std::numeric_limits<int>::max()) {
                b[i]=(*xi)[i][U];
            }else{
                b[i]=(*xi)[i][(*xi)[i].size()-1];
            }
            // Update the area
            areanx = areanx*(b[i] - a[i]); 
        }

        if(subtree){
            area.push_back(areanx);
            mcl.setdeltaarea(areanx, area); // left has same area as parent
            mcr.setdeltaarea(areanx, area); // right has same area as parent
        }else{
            double areal =  areanx;
            double arear =  areanx;
            int L,U;
            int v0, c0;
            v0 = (l->p)->v;
            c0 = (l->p)->c;
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
            (l->p)->rgi(v0,&L,&U);
            
            // Redefine the left and right areas 
            if(L==std::numeric_limits<int>::min()){L = (*xi)[v0][0];}
            if(U==std::numeric_limits<int>::max()){U = (*xi)[v0][(*xi)[v0].size()-1];}
            
            areal = areal*((*xi)[v0][c0]-(*xi)[v0][L])/((*xi)[v0][U]-(*xi)[v0][L]); 
            arear = arear*((*xi)[v0][U]-(*xi)[v0][c0])/((*xi)[v0][U]-(*xi)[v0][L]);
            

            // Add to area vector
            area_bd = area;
            area.push_back(areal); area.push_back(arear);
            area_bd.push_back(areanx);

            // Add areas to suff stats
            mcl.setdeltaarea(areanx, area, area_bd); // left is different than parent
            mcr.setdeltaarea(areanx, area, area_bd); // right is different than parent
        }
    }

}

//--------------------------------------------------
// Overload the birth local_mpigetsuff to account for subtrees. Check the conditions then run the brt version
void mcbrt::local_mpigetsuff(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir)
{
    // Define sil and sir to consider the calibration subtree cases
    local_mpigetsuff_nodecases(nx,sil,sir,true); //true means this is a birth
    // Now that sil and sir are defined to account for calibration subtrees, run the usual brt version
    brt::local_mpigetsuff(nx,v,c,di,sil,sir);
}

//--------------------------------------------------
// Overload the death local_mpigetsuff to account for subtrees. Check the conditions then run the brt version
void mcbrt::local_mpigetsuff(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir)
{
    // Define sil and sir to consider the calibration subtree cases
    local_mpigetsuff_nodecases(l->p,sil,sir,false); //true means this is a death, l->p is the reference node to determine subtree conditions
    // Now that sil and sir are defined to account for calibration subtrees, run the usual brt version
    brt::local_mpigetsuff(l,r,di,sil,sir);
}

//--------------------------------------------------
// Establish the node cases for birth and death
void mcbrt::local_mpigetsuff_nodecases(tree::tree_p n, sinfo& sil, sinfo& sir, bool birthmove)
{
    #ifdef _OPENMPI
    // Only need to initialize the sir and sil for rank 0
    tree::npv uroots,subbnv;
    tree::tree_p subtree;
    if(rank==0){
        // Cast sil and sir to be mcinfo
        mcinfo& mcr=static_cast<mcinfo&>(sir);
        mcinfo& mcl=static_cast<mcinfo&>(sil);
        // Get roots of the subtree(s) in t
        t.getsubtreeroots(uroots, uvec);
        // Check is the node nx in a subtree
        n->nodeinsubtree(uroots,subtree);
        // Now check to see if p is the root of the subtree, if so we can just set subtree to 0. 
        if(n == subtree){subtree = 0;}

        // If not a null pointer and a subtree is used, get bottom nodes and resize mcr subtree info        
        if(subtree){
            subtree->getbots(subbnv);
            // Set subtree_node = true for left and right children
            mcl.subtree_node = true;
            mcr.subtree_node = true;
            //Set mcr as the subtree andn sibling info node
            mcr.sibling_info = true;
            if(birthmove){mcr.resizesubtreeinfo(subbnv.size()-1);}else{mcr.resizesubtreeinfo(subbnv.size()-2);}
        }
    }
#endif
}

//--------------------------------------------------
// subsuff
void mcbrt::subsuff(tree::tree_p nx, tree::npv& bnv, std::vector<sinfo*>& siv)
{
    tree::npv path;
    tree::npv uroots, nxuroots; // Roots to all calibration subtree
    tree::npv obnv, ouroots; // bnv and uroots for orthogonal discrepancy
    tree::tree_p subtree, troot; // subtree pointer root
    tree::tree_p toproot; // toproot for orthogonal delta
    std::vector<double> area; //Area by discrepancy region -- for orthogonal delta
    std::map<tree::tree_cp,double> area_by_node; // for orthogonal discrepancy


    // Set the root of the tree for the rotation. 
    // If nx is in a subtree, then we need to use its subtree root rather than using nx 
    troot = nx;
    local_subsuff_setroot(nx,subtree,troot,uroots);

    bnv.clear();

    troot->getpathtoroot(path);  //path from subtree root troot back to root (troot = nx or subtree if subtree is a pointer above nx)
    troot->getbots(bnv);  //all bots ONLY BELOW node troot!!
    //cout << "troot = " << troot->nid() << "-----" << rank << endl;
    //cout << "tree size = " << troot->treesize() << "-----" << rank << endl;
    //cout << "bnv size = " << bnv.size() << "-----" << rank << endl;
    
#ifdef _OPENMP
    typedef tree::npv::size_type bvsz;
    siv.clear(); //need to setup space threads will add into
    siv.resize(bnv.size());
    for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
#     pragma omp parallel num_threads(tc)
    local_ompsubsuff(*di,troot,path,bnv,siv); //faster if pass di and bnv by value.
#elif _OPENMPI
    diterator diter(di);
    //cout << "bnv.size = " << bnv.size() << " ---- rank = " << rank << endl;
    local_mpisubsuff(diter,troot,path,bnv,siv);
#else
    diterator diter(di);
    local_subsuff(diter,troot,path,bnv,siv);
#endif
    // Now pool information across nodes depending of the calibration subtree status
    local_subsuff_nodecases(troot,subtree,bnv,siv);

    // Orthogonal discrepancy - get scales
    if(orth_delta){
        // Get the very top root of the tree
        toproot = path[path.size()-1];
        // Get its bottom nodes
        toproot->getbots(obnv);
        // Get subtree information
        toproot->getsubtreeroots(ouroots, uvec);
        // Get the orthogonal discrepancy scales
        get_orthogonal_scales_rot(obnv, ouroots, area, area_by_node);
        // Now map each siv element to an area_by_node
        for(size_t i=0;i<bnv.size();i++){
            mcinfo& mci=static_cast<mcinfo&>(*(siv[i]));
            mci.setdeltaarea(area_by_node[bnv[i]],area);
        }  
    }
}

//--------------------------------------------------
//local_subsuff_setroot
void mcbrt::local_subsuff_setroot(tree::tree_p nx,tree::tree_p &subtree,tree::tree_p &troot,tree::npv &uroots){
    // Get calibration subtree information
    t.getsubtreeroots(uroots, uvec);
    //cout << "uroots size = " << uroots.size() << endl;
    if(uroots.size()>0){
        // Check is the node nx in a subtree
        nx->nodeinsubtree(uroots,subtree); // see if nx is in a subtree...sets subtree as a tree pointer or null pointer
    }else{
        // set null pointer, no subtree
        subtree = 0;
    }
    // if nx is in a subtree, then set the subsuff tree root to be the subtree root, else assign troot = nx
    if(subtree){
        troot = subtree;
    }else{
        troot = nx;
    }
    //cout << "subtree = " << subtree << "--- rank =" << rank << endl;
}

//--------------------------------------------------
//local_subsuff
/*
void mcbrt::local_subsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv)
{
    tree::npv uroots, nxuroots; // Roots to all calibration subtree
    tree::tree_p subtree; // subtree pointer root

    // Get calibration subtree information
    t.getsubtreeroots(uroots, uvec);
    // Check is the node nx in a subtree
    nx->nodeinsubtree(uroots,subtree); // could make more efficeint by utilizing the path we read in
    
    // if nx isn't in a calibration subtree or starts one, then check to see if it contains one (or many)
    if(!subtree){
        // Get subtree roots below nx
        nx->getsubtreeroots(nxuroots,uvec);
    }

    // Check possible conditions
    if(!subtree && nxuroots.size()==0){
        // If no subtree is involved in the rotation, use the brt::local_subsuff definiition
        brt::local_subsuff(diter, nx, path, bnv, siv);
    }else if(!subtree && nxuroots.size()>0){
        // nx is not in calibration subtree nor does not split on calibration parameter
        // But, a calibration subtree(s) are created below nx 
        local_subsuff_subtree(nxuroots, diter, nx, path, bnv, siv);
    }else{
        // Either nx creates the calibration subtree or it is in one already
        local_subsuff_subtree(subtree, diter, nx, path, bnv, siv);
    } 
}
*/

//--------------------------------------------------
//local_getsuff_subtree (1) -- used when a subtree begins at or above the rotation node
void mcbrt::local_subsuff_subtree(std::vector<sinfo*>& siv)
{
    // Cast suff stats to mcinfo -- this usually happens in add_obs_to_suff but we need it here to set 
    // Initialize mcinfo vector by casting siv to mcinfo
    std::vector<mcinfo*> mcv(siv.size()); // mcinfo vector used for subtrees
    for(size_t i=0;i<siv.size();i++){
        mcv[i]=static_cast<mcinfo*>(siv[i]);
        mcv[i]->subtree_node = true;
    }
    
    // Compile suff stats across the calibration subtree--WLOG, store everything in the first node of this vector
    std::vector<mcinfo*> mcv_not0(mcv.begin()+1,mcv.end()); // vector excluding the first element
    mcv[0]->setsubtreeinfo(mcv_not0);
    //cout << "Pool subtree info, mcv[0] = ..." << endl;
    //mcv[0]->print();
}

//--------------------------------------------------
//local_getsuff_subtree (2) -- used when a subtree or subtrees begin below the rotation node
void mcbrt::local_subsuff_subtree(tree::npv nxuroots, tree::tree_p nx, tree::npv& bnv, std::vector<sinfo*>& siv)
{
    typedef tree::npv::size_type bvsz;
    std::vector<mcinfo*> mcv(bnv.size()); // mcinfo vector used for subtrees
    tree::tree_p subtree;

    // Define bottom node vector map, unode map for calibration subtreen, and clear siv
    std::map<tree::tree_cp,std::vector<size_t>> unmap;
    // Initialize suff stat vectors
    for(bvsz i=0;i<bnv.size();i++) { 
        mcv[i] = static_cast<mcinfo*>(siv[i]);
        bnv[i]->nodeinsubtree(nxuroots,subtree);
        // Append this nodes id to its corresponding unodemap if its in a subtree
        if(subtree){
            // This node is in a calibration subtree
            mcv[i]->subtree_node = true;
            unmap[subtree].push_back(i); // append the id to the utree id map -- keeps track of which nodes belong to which subtrees 
        }
    }
    // Compile suff stats across the calibration subtree(s)
    std::vector<mcinfo*> mcvtemp;
    for(size_t i=0;i<unmap.size();i++){
        // Get the first index of the nodes in this calibration subtree, then set subtree info 
        size_t uind = unmap[nxuroots[i]][0]; 
        mcvtemp.clear();

        // Populate the temp vector with the subtree info on other nodes but the first
        for(size_t j=1;j<unmap[nxuroots[i]].size();j++){mcvtemp.push_back(mcv[unmap[nxuroots[i]][j]]);} 
        mcv[uind]->setsubtreeinfo(mcvtemp); // Load the subtree info on the representative node of this subtree
    }
}

//--------------------------------------------------
//local_mpisubsuff_resize_vecs
// This is a helper function to resize the bottom node and suff stat vectors for rank 0 in local_mpisubsuff
// Also sets which one will be in charge of the subtree info 
// A lot of similar code to the above local_subsuff cases, try to clean up later
void mcbrt::local_subsuff_nodecases(tree::tree_p nx, tree::tree_p subtree, tree::npv& bnv, std::vector<sinfo*>& siv){
    tree::npv nxuroots; // Roots to all calibration subtree
    // if nx isn't in a calibration subtree or starts one, then check to see if it contains one (or many)
    if(!subtree){
        // Get subtree roots below nx
        nx->getsubtreeroots(nxuroots,uvec); 
    }
    // Now check conditions for how to pool the information across nodes in a common subtree
    if(!subtree && nxuroots.size()>0){
        // If nx is not in a subtree, but nodes below nx are in subtree(s)
        local_subsuff_subtree(nxuroots,nx,bnv,siv);
    }else if(subtree){
        // Either nx creates the calibration subtree or it is in one already
        local_subsuff_subtree(siv);        
    }else{
        // This set of suff stats is not associated with a claibration subtree -- no need to pool information    
    }
}


//--------------------------------------------------
//local_mpisubsuff -- overload this function from brt to account for the three different cases in rotation on rank 0
/*
void mcbrt::local_mpisubsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   if(rank==0){
      // Resize bnv and siv depending on the calibration subtree status, also assign which nodes will hold subtree info
      local_mpisubsuff_nodecases(nx,bnv,siv);
      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }else{
      local_subsuff(diter,nx,path,bnv,siv); // handles the three cases when off of rank 0
      local_mpi_reduce_allsuff(siv); 
   }
#endif
}
*/

//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
void mcbrt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
    mcinfo& msil=static_cast<mcinfo&>(sil);
    mcinfo& msir=static_cast<mcinfo&>(sir);
    int buffer_size = SIZE_UINT6*10; 
    if(rank==0) { // MPI receive all the answers from the slaves
        MPI_Status status;
        mcinfo& tsil = (mcinfo&) *newsinfo();
        mcinfo& tsir = (mcinfo&) *newsinfo();
        int position=0;
        unsigned int ln,rn;
        for(size_t i=1; i<=(size_t)tc; i++) {
            // Pass subtree information if present
            // This should only apply to the right node
            size_t ns = 0;
            ns = tsir.subtree_sumyw.size();
            double sbt_sumyw_array[ns], sbt_sumzw_array[ns];
            double sbt_sumwf_array[ns], sbt_sumwc_array[ns];
            unsigned int sbt_nf_array[ns];
            
            if(msir.subtree_info){
                // Cast vectors to array
                std::copy(tsir.subtree_sumyw.begin(),tsir.subtree_sumyw.end(),sbt_sumyw_array);
                std::copy(tsir.subtree_sumzw.begin(),tsir.subtree_sumzw.end(),sbt_sumzw_array);
                std::copy(tsir.subtree_sumwf.begin(),tsir.subtree_sumwf.end(),sbt_sumwf_array);
                std::copy(tsir.subtree_sumwc.begin(),tsir.subtree_sumwc.end(),sbt_sumwc_array);
                std::copy(tsir.subtree_nf.begin(),tsir.subtree_nf.end(),sbt_nf_array);
                tsir.subtree_info = true;
            }
            
            // Ensure tsir and tsil are correctly marked as subtree nodes if applicable
            if(msir.subtree_node){tsir.subtree_node=true;}
            if(msil.subtree_node){tsil.subtree_node=true;}

            char buffer[buffer_size];    
            position=0;
            MPI_Recv(buffer,buffer_size,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
            MPI_Unpack(buffer,buffer_size,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsil.nf,1,MPI_UNSIGNED,MPI_COMM_WORLD); // May not need to pass
            MPI_Unpack(buffer,buffer_size,&position,&tsir.nf,1,MPI_UNSIGNED,MPI_COMM_WORLD); // May not need to pass
            MPI_Unpack(buffer,buffer_size,&position,&tsil.sumwf,1,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsir.sumwf,1,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsil.sumwc,1,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsir.sumwc,1,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsil.sumyw,1,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsir.sumyw,1,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsil.sumzw,1,MPI_DOUBLE,MPI_COMM_WORLD);
            MPI_Unpack(buffer,buffer_size,&position,&tsir.sumzw,1,MPI_DOUBLE,MPI_COMM_WORLD);
            // Now unpack the subtree information and cast back to std::vector 
            if(msir.subtree_info){
                MPI_Unpack(buffer,buffer_size,&position,&sbt_sumyw_array,ns,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&sbt_sumzw_array,ns,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&sbt_sumwf_array,ns,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&sbt_sumwc_array,ns,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&sbt_nf_array,ns,MPI_UNSIGNED,MPI_COMM_WORLD);

                for(size_t j=0;j<ns;j++){tsir.subtree_sumyw.push_back(sbt_sumyw_array[j]);}
                for(size_t j=0;j<ns;j++){tsir.subtree_sumzw.push_back(sbt_sumzw_array[j]);}
                for(size_t j=0;j<ns;j++){tsir.subtree_sumwf.push_back(sbt_sumwf_array[j]);}
                for(size_t j=0;j<ns;j++){tsir.subtree_sumwc.push_back(sbt_sumwc_array[j]);}
                for(size_t j=0;j<ns;j++){tsir.subtree_nf.push_back(sbt_nf_array[j]);}
            }
            // Unpack sibling info for the right node (it should have sibling info, otherwise there is a problem)
            // God willing, you'll avoid having to pass this sibling info
            if(msir.sibling_info){
                MPI_Unpack(buffer,buffer_size,&position,&tsir.sibling_sumyw,1,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&tsir.sibling_sumzw,1,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&tsir.sibling_sumwf,1,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&tsir.sibling_sumwc,1,MPI_DOUBLE,MPI_COMM_WORLD);
                MPI_Unpack(buffer,buffer_size,&position,&tsir.sibling_nf,1,MPI_UNSIGNED,MPI_COMM_WORLD);
                tsir.sibling_info = true;
            }
            tsil.n=(size_t)ln;
            tsir.n=(size_t)rn;
            msil+=tsil;
            msir+=tsir;
        }
        delete &tsil;
        delete &tsir;
    }
    else // MPI send all the answers to root
    {
        int position=0;  
        unsigned int ln,rn;

        size_t ns = 0;
        ns = msir.subtree_sumyw.size();
        double sbt_sumyw_array[ns], sbt_sumzw_array[ns];
        double sbt_sumwf_array[ns], sbt_sumwc_array[ns];
        unsigned int sbt_nf_array[ns];
        if(msir.subtree_info){
            // Cast vectors to array
            std::copy(msir.subtree_sumyw.begin(),msir.subtree_sumyw.end(),sbt_sumyw_array);
            std::copy(msir.subtree_sumzw.begin(),msir.subtree_sumzw.end(),sbt_sumzw_array);
            std::copy(msir.subtree_sumwf.begin(),msir.subtree_sumwf.end(),sbt_sumwf_array);
            std::copy(msir.subtree_sumwc.begin(),msir.subtree_sumwc.end(),sbt_sumwc_array);
            std::copy(msir.subtree_nf.begin(),msir.subtree_nf.end(),sbt_nf_array); 
        }
        
        char buffer[buffer_size];
        ln=(unsigned int)msil.n;
        rn=(unsigned int)msir.n;
        MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msil.nf,1,MPI_UNSIGNED,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msir.nf,1,MPI_UNSIGNED,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msil.sumwf,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msir.sumwf,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msil.sumwc,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msir.sumwc,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msil.sumyw,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msir.sumyw,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msil.sumzw,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        MPI_Pack(&msir.sumzw,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
        // Now Pack the subtree information 
        if(msir.subtree_info){
            MPI_Pack(&sbt_sumyw_array,ns,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
            MPI_Pack(&sbt_sumzw_array,ns,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
            MPI_Pack(&sbt_sumwf_array,ns,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
            MPI_Pack(&sbt_sumwc_array,ns,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
            MPI_Pack(&sbt_nf_array,ns,MPI_UNSIGNED,buffer,buffer_size,&position,MPI_COMM_WORLD);
        }
        // Pack the sibling info for the right node (if should have sibling info, otherwise there is a problem)
        if(msir.sibling_info){
            MPI_Pack(&msir.sibling_sumyw,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
            MPI_Pack(&msir.sibling_sumzw,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
            MPI_Pack(&msir.sibling_sumwf,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
            MPI_Pack(&msir.sibling_sumwc,1,MPI_DOUBLE,buffer,buffer_size,&position,MPI_COMM_WORLD);
            MPI_Pack(&msir.sibling_nf,1,MPI_UNSIGNED,buffer,buffer_size,&position,MPI_COMM_WORLD);
        }
        MPI_Send(buffer,buffer_size,MPI_PACKED,0,0,MPI_COMM_WORLD);     
    }
#endif   
}

//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void mcbrt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
    unsigned int nvec[siv.size()];
    unsigned int nfvec[siv.size()];
    double sumywvec[siv.size()];
    double sumzwvec[siv.size()];
    double sumwfvec[siv.size()];
    double sumwcvec[siv.size()];
    /*
    // Subtree versions
    std::vector<double> sb_sumyw_vec;
    std::vector<double> sb_sumzw_vec;
    std::vector<double> sb_sumwf_vec;
    std::vector<double> sb_sumwc_vec;
    */
    for(size_t i=0;i<siv.size();i++) { // on root node, these should be 0 because of newsinfo().
        mcinfo* msi=static_cast<mcinfo*>(siv[i]);
        nvec[i]=(unsigned int)msi->n;    // cast to int
        nfvec[i]=(unsigned int)msi->nf;
        sumywvec[i]=msi->sumyw;
        sumzwvec[i]=msi->sumzw;
        sumwfvec[i]=msi->sumwf;
        sumwcvec[i]=msi->sumwc;

        // Populate vectors of subtree info (total subtree info across all nodes..
        //if there are multiple subtrees, then we can append to the vector and separate later)
        /*
        if(mis->subtree_info){
            sb_sumyw_vec.insert(std::end(sb_sumyw_vec),std::begin(mis->subtree_sumyw),std::end(mis->subtree_sumyw));
            sb_sumzw_vec.insert(std::end(sb_sumzw_vec),std::begin(mis->subtree_sumzw),std::end(mis->subtree_sumzw));
            sb_sumwf_vec.insert(std::end(sb_sumwf_vec),std::begin(mis->subtree_sumwf),std::end(mis->subtree_sumwf));
            sb_sumwc_vec.insert(std::end(sb_sumwc_vec),std::begin(mis->subtree_sumwc),std::end(mis->subtree_sumwc));        
        }
        */
    }

    // Now initialize the subtree suff stat mpi arrays
    /*
    size_t sbtvs = sb_sumyw_vec.size():
    double subtree_sumyw[sbtvs];
    double subtree_sumzw[sbtvs];
    double subtree_sumwf[sbtvs];
    double subtree_sumwc[sbtvs];
    if(sbtvs > 0){
        copy(sb_sumyw_vec.begin(),sb_sumyw_vec.end(), subtree_sumyw);
        copy(sb_sumzw_vec.begin(),sb_sumzw_vec.end(), subtree_sumzw);
        copy(sb_sumwf_vec.begin(),sb_sumwf_vec.end(), subtree_sumwf);
        copy(sb_sumwc_vec.begin(),sb_sumwc_vec.end(), subtree_sumwc);
    }
    */

    if(rank==0) {
        MPI_Status status;
        unsigned int tempnvec[siv.size()];
        unsigned int tempnfvec[siv.size()];
        double tempsumywvec[siv.size()];
        double tempsumzwvec[siv.size()];
        double tempsumwfvec[siv.size()];
        double tempsumwcvec[siv.size()];
        
        //cout << "siv.size() (rank) = " << siv.size() << "----" << rank << endl;
        // receive nvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempnvec,siv.size(),MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++){
                nvec[j]+=tempnvec[j];
                //cout << "nvec = " << nvec[j] << endl; 
            }
        }
        MPI_Request *request=new MPI_Request[tc];

        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        
        // cast back to mci
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->n=(size_t)nvec[i];    // cast back to size_t
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;


        // receive nfvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempnfvec,siv.size(),MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++){
                nfvec[j]+=tempnfvec[j];
                //cout << "nfvec = " << nfvec[j] << endl; 
            }
        }
        request=new MPI_Request[tc];

        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&nfvec,siv.size(),MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        
        // cast back to mci
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->nf=(size_t)nfvec[i];    // cast back to size_t
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;


        // receive sumywvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumywvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++){
                sumywvec[j]+=tempsumywvec[j];
                //cout << "sumyw = " << sumywvec[j] << endl;
            }
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumywvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->sumyw=sumywvec[i];
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;
        
        // receive sumzwvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumzwvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++){
                sumzwvec[j]+=tempsumzwvec[j];
                //cout << "sumzw = " << sumzwvec[j] << endl;
            }
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumzwvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->sumzw=sumzwvec[i];
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        // receive sumwfvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumwfvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++){
                sumwfvec[j]+=tempsumwfvec[j];
                //cout << "sumwf = " << sumwfvec[j] << endl;
            }
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumwfvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->sumwf=sumwfvec[i];
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        // receive sumwcvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumwcvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++){
                sumwcvec[j]+=tempsumwcvec[j];
                //cout << "sumwc = " << sumwcvec[j] << endl;
            }
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumwcvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->sumwc=sumwcvec[i];
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;
    
    } else {

        MPI_Request *request=new MPI_Request;
        MPI_Status status;
        //cout << "siv.size() (rank) = " << siv.size() << "----" << rank << endl;
        // send/recv nvec      
        MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&nvec,siv.size(),MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send nfvec, update nvec, receive nfwvec
        request=new MPI_Request;
        MPI_Isend(&nfvec,siv.size(),MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->n=(size_t)nvec[i];    // cast back to size_t
            //cout << "nvec = " << nvec[i] << endl;
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&nfvec,siv.size(),MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumywvec, update nfvec, receive sumywvec
        request=new MPI_Request;
        MPI_Isend(&sumywvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->nf=(size_t)nfvec[i];    // cast back to size_t
            //cout << "nfvec = " << nfvec[i] << endl;
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumywvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumzwvec, update sumywvec, receive sumzwvec
        request=new MPI_Request;
        MPI_Isend(&sumzwvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->sumyw=sumywvec[i];
            //cout << "sumywvec = " << sumywvec[i] << endl;
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumzwvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumwfvec, update sumzwvec, receive sumwfvec
        request=new MPI_Request;
        MPI_Isend(&sumwfvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->sumzw=sumzwvec[i];
            //cout << "sumzwvec = " << sumzwvec[i] << endl;
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumwfvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumwcvec, update sumwfvec, receive sumwcvec
        request=new MPI_Request;
        MPI_Isend(&sumwcvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->sumwf=sumwfvec[i];
            //cout << "sumwfvec = " << sumwfvec[i] << endl;
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumwcvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // update sumwcvec
        // cast back to msi
        for(size_t i=0;i<siv.size();i++) {
            mcinfo* msi=static_cast<mcinfo*>(siv[i]);
            msi->sumwc=sumwcvec[i];
            //cout << "sumwcvec = " << sumwcvec[i] << endl;
        }
    }
#endif
}

//--------------------------------------------------
//Print mcbrt object
void mcbrt::pr_vec()
{
   std::cout << "***** mcbrt object:\n";
   std::cout << "Conditioning info:" << std::endl;
   std::cout << "   mean:   tau1 =" << ci.tau1 << std::endl;
   std::cout << "   mean:   tau2 =" << ci.tau2 << std::endl;
   std::cout << "   mean:   mu1 =" << ci.mu1 << std::endl;
   std::cout << "   mean:   mu2 =" << ci.mu2 << std::endl;
   if(!ci.sigma)
     std::cout << "         sigma=[]" << std::endl;
   else
     std::cout << "         sigma=[" << ci.sigma[0] << ",...," << ci.sigma[di->n-1] << "]" << std::endl;
   brt::pr_vec();
}

//--------------------------------------------------
// Gradient of log KL-based density 
std::vector<double> mcbrt::klgradient(size_t nf, std::vector<double> gradh,std::vector<double> &xgrad, std::vector<size_t> ucols, std::vector<double> uvals){
    std::vector<double> grad(ucols.size(),0);
    size_t pu = ucols.size();
    size_t p = di->p;
    size_t uc;
    std::vector<double> etadesign {1.0, 0.0};
    std::vector<double> eta0(nf,0);
    mxd etamx(nf,2);

    // Create digrad
    dinfo digrad;
    double *fcgrad = new double[nf];
    digrad.n=0;digrad.p=p,digrad.x = NULL;digrad.y=NULL;digrad.tc=tc;
    if(rank>0){digrad.n=nf; digrad.x = &xgrad[0]; digrad.y = &fcgrad[0];}

    // initialize etamx (check nf>0 condition to handle rank 0 on mpi)
    if(nf>0){
        etamx.rowwise() = Eigen::Map<Eigen::RowVectorXd>(etadesign.data(), 2);
    }

    // Get initial predictions at the input xgrad
    if(rank>0){
        predict_vec(&digrad,&etamx);
    }    
    for(size_t i=0;i<nf;i++){eta0[i] = fcgrad[i];}
    
    // Get the finite differences
    for(size_t j=0;j<pu;j++){
        // Update x with the stepsize
        uc = ucols[j];
        if(j==0){
            for(size_t i=0;i<nf;i++){xgrad[i*p + uc] = uvals[j] + gradh[j];}    
        }else{
            for(size_t i=0;i<nf;i++){
                xgrad[i*p + ucols[j-1]] = uvals[j-1]; // reset previous value
                xgrad[i*p + uc] = uvals[j] + gradh[j]; // apply the increase 
            }
        }
            
        if(rank>0){
            predict_vec(&digrad,&etamx);
        }

        for(size_t i=0;i<nf;i++){
            //grad[j] += (f(i)-eta(i))*(fcgrad[i]-eta(i))/h;
            grad[j] += (f(i)-eta0[i])*(fcgrad[i]-eta0[i])/(gradh[j]*ci.sigma[i]*ci.sigma[i]); // average grad
        }   
    }

    // return the gradient
    return grad;
}

//--------------------------------------------------
// get orthogonal scales -- for orthogonal discrepancy
std::vector<double> mcbrt::get_orthogonal_scales(tree::npv bnv, tree::npv uroots, std::map<tree::tree_cp, std::vector<size_t>> unmap, std::vector<size_t> xnodes){
    // Set scales for the xnodes
    int B = xnodes.size() + uroots.size();
    std::vector<std::vector<double> > a(uvec[0],std::vector<double>(B));
    std::vector<std::vector<double> > b(uvec[0],std::vector<double>(B));
    std::vector<double> area(B,1);
    double atemp = 1.0;

    // Get the rectangles with respect to x-space (assume u predictors come after all x predictors)
    //cout << "uvec[0] = " << uvec[0] << endl;
    //cout << "xnodes.size() = " << xnodes.size() << endl;
    //cout << "unmap.size() = " << unmap.size() << endl;
    for(size_t i=0;i<uvec[0];i++){
        for(size_t j = 0; j<xnodes.size();j++){
            // reset upper and lower bounds
            int L,U;
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();

            // Get the interval [L,U] indexes
            size_t ind = xnodes[j];
            bnv[ind]->rgi(i,&L,&U);

            // Now we have the interval endpoints, put corresponding values in a,b matrices.
            if(L!=std::numeric_limits<int>::min()){ 
                a[i][j]=(*xi)[i][L];
            }else{
                a[i][j]=(*xi)[i][0];
            }
            if(U!=std::numeric_limits<int>::max()) {
                b[i][j]=(*xi)[i][U];
            }else{
                b[i][j]=(*xi)[i][(*xi)[i].size()-1];
            }
            // Update the area
            //cout << "area[j] = " << area[j] << endl;
            //cout << "a[i][j] = " << a[i][j] << endl;
            //cout << "b[i][j] = " << b[i][j] << endl;
            area[j] = area[j]*(b[i][j] - a[i][j]); 
        }

        // Set scales for the subtrees
        for(size_t j = 0; j<uroots.size();j++){
            // reset upper and lower bounds
            int L,U;
            size_t l = j + xnodes.size();
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();

            // Get the interval [L,U] indexes
            size_t ind = unmap[uroots[j]][0];
            bnv[ind]->rgi(i,&L,&U);

            // Now we have the interval endpoints, put corresponding values in a,b matrices.
            if(L!=std::numeric_limits<int>::min()){ 
                a[i][l]=(*xi)[i][L];
            }else{
                a[i][l]=(*xi)[i][0];
            }
            if(U!=std::numeric_limits<int>::max()) {
                b[i][l]=(*xi)[i][U];
            }else{
                b[i][l]=(*xi)[i][(*xi)[i].size()-1];
            }
            // Update the area
            area[l] = area[l]*(b[i][l] - a[i][l]); 
        }
    }

    // Condition for when only a root node in the tree
    if(B == 0){ 
        for(size_t i = 0; i<uvec[0];i++){
            atemp = atemp*((*xi)[i][(*xi)[i].size()-1] - (*xi)[i][0]);
        }
        area.push_back(atemp);
    }

    // Return area
    return(area);
}


//--------------------------------------------------
// Set orthogonal scales for birth and death
std::vector<double> mcbrt::get_orthogonal_scales_bd(tree::npv obnv, tree::npv uroots, tree::tree_p nx){
    std::vector<size_t> xnodes;
    tree::tree_p subtree;
    std::map<tree::tree_cp,std::vector<size_t>> unmap; // subtree map
    std::vector<double> area;

    // Check is the node in bnv is in a subtree
    for(size_t i=0;i<obnv.size();i++){
        subtree = 0;
        // Setup to for the area calculation
        if(obnv[i] != nx){
            obnv[i]->nodeinsubtree(uroots,subtree);
            if(subtree){
                // This node is in a subtree, append i to the corresponding map
                unmap[subtree].push_back(i);
            }else{
                // Not in a subtree
                xnodes.push_back(i);
            }
        }
    }
        
    // Now get areas
    area = get_orthogonal_scales(obnv, uroots, unmap, xnodes);
    return(area);
}


//--------------------------------------------------
// Orthogonal Scales for draw step
std::vector<std::vector<double>> mcbrt::get_orthogonal_scales_draw(tree::npv bnv, tree::npv uroots, 
                                                                std::map<tree::tree_cp,std::vector<size_t>> unmap, 
                                                                std::vector<size_t> xnodes)
{
    std::vector<double> area;
    std::vector<double> area_by_node(bnv.size());
    std::vector<std::vector<double>> area_out(2);
    size_t ind, aind;

    // Get the area vector -- unique areas
    area = get_orthogonal_scales(bnv, uroots, unmap, xnodes);

    // Expand area to get the areas for each node
    for(size_t j = 0; j<xnodes.size();j++){
        // Get the interval [L,U] indexes
        ind = xnodes[j];
        area_by_node[ind] = area[j];
    }

    // Set scales for the subtrees
    for(size_t j=0; j<uroots.size();j++){
        aind = xnodes.size()+j;
        for(size_t l=0; l<unmap[uroots[j]].size();l++){
            ind = unmap[uroots[j]][l];
            area_by_node[ind] = area[aind];
        }
    }

    // Set the area out vector
    area_out[0] = area;
    area_out[1] = area_by_node;
    
    // Return area
    return(area_out);                                                    
}

//--------------------------------------------------
// Get orthogonal scales -- rotate and perturb
void mcbrt::get_orthogonal_scales_rot(tree::npv obnv, tree::npv uroots, std::vector<double> &area,std::map<tree::tree_cp,double> &area_by_node)
{
    std::vector<size_t> xnodes;
    tree::tree_p subtree;
    std::map<tree::tree_cp,std::vector<size_t>> unmap; // subtree map
    size_t aind, ind;

    // Check is the node in bnv is in a subtree
    for(size_t i=0;i<obnv.size();i++){
        subtree = 0;
        obnv[i]->nodeinsubtree(uroots,subtree);
        if(subtree){
            // This node is in a subtree, append i to the corresponding map
            unmap[subtree].push_back(i);
        }else{
            // Not in a subtree
            xnodes.push_back(i);
        }
    }

    // Now call the original set_orthogonal_scales
    area = get_orthogonal_scales(obnv, uroots, unmap, xnodes);

    //convert area vector into a map for area by node
    for(size_t j = 0; j<xnodes.size();j++){
        // Get the interval [L,U] indexes
        ind = xnodes[j];
        area_by_node[obnv[ind]] = area[j];
    }

    // Set scales for the subtrees
    for(size_t j=0; j<uroots.size();j++){
        aind = xnodes.size()+j;
        for(size_t l=0; l<unmap[uroots[j]].size();l++){
            ind = unmap[uroots[j]][l];
            area_by_node[obnv[ind]] = area[aind];
        }
    }

} 


//--------------------------------------------------
// Set orthogonal scales without xroots -- adds the additional step of computing xnodes vector
std::vector<double> mcbrt::get_orthogonal_scales(tree::npv obnv, tree::npv uroots){
    std::vector<size_t> xnodes;
    tree::tree_p subtree;
    std::map<tree::tree_cp,std::vector<size_t>> unmap; // subtree map
    std::vector<double> area;

    // Check is the node in bnv is in a subtree
    for(size_t i=0;i<obnv.size();i++){
        subtree = 0;
        obnv[i]->nodeinsubtree(uroots,subtree);
        if(subtree){
            // This node is in a subtree, append i to the corresponding map
            unmap[subtree].push_back(i);
        }else{
            // Not in a subtree
            xnodes.push_back(i);
        }
    }

    // Now call the original set_orthogonal_scales
    area = get_orthogonal_scales(obnv, uroots, unmap, xnodes);
    return(area);
}
