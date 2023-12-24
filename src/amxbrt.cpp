#include "amxbrt.h"
#include "brtfuns.h"
#include <iostream>
#include <map>
#include <vector>

using std::cout;
using std::endl;

//--------------------------------------------------
//a single iteration of the MCMC for brt model
void amxbrt::drawvec(rn& gen){
    for(size_t j=0;j<m;j++) {
        //Uses operators defined in dinfo on y to compute the jth residual Rij
        *divec[j] = *di;
        *divec[j]-= *getf(); //Yi - sum_k g(xi, Tk, Mk) 
        *divec[j]+= *mb[j].getf(); //Computes the jth residual in BART model

        //Draw parameter vector in the jth tree
        mb[j].drawvec(gen);
        
        // Update the in-sample predicted vector
        setf_vec();

        // Update the in-sample residual vector
        setr_vec();
    }
    // overall statistics from the subtrees.  Need to divide by m*N to get
    // useful numbers after the MCMC is done.
    if(mi.dostats) {
    resetstats();
    for(size_t j=0;j<m;j++){
        mb[j].addstats(mi.varcount,&mi.tavgd,&mi.tmaxd,&mi.tmind);
    }
  }
}

//--------------------------------------------------
//slave controller for draw when using MPI
void amxbrt::drawvec_mpislave(rn& gen){
    for(size_t j=0;j<m;j++) {
        //Get the jth resiudal
        *divec[j]= *di;
        *divec[j]-= *getf();
        *divec[j]+= *mb[j].getf();

        // do the draw for jth component
        mb[j].drawvec_mpislave(gen);
        
        // Update the in-sample predicted vector
        setf_vec();

        // Update the in-sample residual vector
        setr_vec();
    }
}

//--------------------------------------------------
//adapt the proposal widths for perturb proposals,
//bd or rot proposals and b or d proposals.
void amxbrt::adapt(){
  for(size_t j=0;j<m;j++) {
#ifndef SILENT
    cout << "\nAdapt ambrt[" << j << "]:";
#endif
    mb[j].adapt();
  }
}

//--------------------------------------------------
//setdata for amxbrt
void amxbrt::setdata_vec(dinfo *di) {
    this->di=di;
        
    // initialize notjsigmavs.
    for(size_t j=0;j<m;j++){
        notjmus[j].resize(this->di->n,0.0);
    }
    for(size_t j=0;j<m;j++){
        for(size_t i=0;i<di->n;i++){
            //notjmus[j][i]=this->di->y[i]/((double)m);
            notjmus[j][i]=this->di->y[i];
        }
    }
    for(size_t j=0;j<m;j++){
        divec[j]=new dinfo(this->di->p,this->di->n,this->di->x,&notjmus[j][0],this->di->tc); //constructing a new dinfo with notjmus[j][0] as the y value 
    }
    /*
    diterator diter(divec[0]);
    for(;diter<diter.until();diter++){
        cout << diter.getx() << " ------- " << *diter << " ------- " << diter.gety() << endl;
    }
    */
    // each mb[j]'s data is the appropriate row in notjmus
    for(size_t j=0;j<m;j++){
        mb[j].setdata_vec(divec[j]); //setdata_vec is a method of mb[j] which is a member of mxbrt class. This is different than setdata_vec in mxbrt
    }
    resid.resize(di->n);
    yhat.resize(di->n);
    setf_vec();
    setr_vec();
}

//--------------------------------------------------
//set vector of predicted values for psbrt model
void amxbrt::local_setf_vec(diterator& diter){
   for(;diter<diter.until();diter++){
      yhat[*diter]=0.0;
      for(size_t j=0;j<m;j++)
        yhat[*diter]+=mb[j].f(*diter); //sum of trees - add the fitted value from each tree
   }
}

//--------------------------------------------------
//set vector of residuals for psbrt model
void amxbrt::local_setr_vec(diterator& diter){
   for(;diter<diter.until();diter++) {
      resid[*diter]=di->y[*diter]-f(*diter);
   }
}

//--------------------------------------------------
//predict the response at the (npred x p) input matrix *x
//Note: the result appears in *dipred.y.
void amxbrt::local_predict_vec(diterator& diter, finfo& fipred){
    tree::tree_p bn;
    double temp;
    vxd thetavec_temp(k); 
    for(;diter<diter.until();diter++) {
        //Initialize to zero 
        temp = 0;
        thetavec_temp = vxd::Zero(k);
        
        for(size_t j=0;j<m;j++) {
            bn = mb[j].t.bn(diter.getxp(),*xi);
            thetavec_temp = bn->getthetavec();
            temp = temp + fipred.row(*diter)*thetavec_temp;
        }
        //std::cout << temp << std::endl;
        diter.sety(temp);
    }
}

//--------------------------------------------------
//extract model weights
void amxbrt::local_predict_thetavec(diterator& diter, mxd& wts){
    tree::tree_p bn;
    vxd thetavec_temp(k);
    for(;diter<diter.until();diter++) {
        //Initialize to zero 
        thetavec_temp = vxd::Zero(k);
        //Get sum of trees for the model weights
        for(size_t j=0;j<m;j++) {
            bn = mb[j].t.bn(diter.getxp(),*xi);
            thetavec_temp = thetavec_temp + bn->getthetavec();
        }
        wts.col(*diter) = thetavec_temp; //sets the thetavec to be the ith column of the wts eigen matrix. 
    }
}

//--------------------------------------------------
// Project each theta vector onto the simplex
void amxbrt::project_thetavec(std::vector<double> &v, std::vector<double>& vstar){
    // Scale v by m
    //std::vector<double> v0;
    //for(size_t i=0;i<k;i++){v0.push_back((*v)[i]*m);}
    
    // Get projection and rescale by 1/m
    brt::project_thetavec(v,vstar);
    //for(size_t i=0;i<k;i++){vstar[i] = vstar[i]/m;}
    for(size_t i=0;i<k;i++){vstar[i] = vstar[i];}
}

//--------------------------------------------------
//extract terminal node parameters for a specific point -- remove
void amxbrt::local_get_mix_theta(diterator& diter, mxd& wts){
    tree::tree_p bn;
    vxd thetavec_temp(k);
    bool enter = true;
    for(;diter<diter.until();diter++) {
        //Initialize to zero 
        thetavec_temp = vxd::Zero(k);
        //Get sum of trees for the model weights
        if(enter){
            for(size_t j=0;j<m;j++) {
                bn = mb[j].t.bn(diter.getxp(),*xi);
                thetavec_temp = bn->getthetavec();
                wts.col(j) = thetavec_temp; //sets the thetavec to be the ith column of the wts eigen matrix.
            }
            enter = false;
        }
        
    }
}

//--------------------------------------------------
//Local Save tree
void amxbrt::local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
    std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta){
    size_t indx=iter*m;
    for(size_t i=(indx+(size_t)beg);i<(indx+(size_t)end);i++) {
        nn[i]=mb[i-indx].t.treesize();
        id[i].resize(nn[i]);
        v[i].resize(nn[i]);
        c[i].resize(nn[i]);
        theta[i].resize(k*nn[i]);
        mb[i-indx].t.treetovec(&id[i][0],&v[i][0],&c[i][0],&theta[i][0],k);
    }
}

//--------------------------------------------------
// Local saving with random hyperparameters
void amxbrt::local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
    std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta,
    std::vector<std::vector<double> >& hyper){
    size_t indx=iter*m;
    for(size_t i=(indx+(size_t)beg);i<(indx+(size_t)end);i++) {
        nn[i]=mb[i-indx].t.treesize();
        id[i].resize(nn[i]);
        v[i].resize(nn[i]);
        c[i].resize(nn[i]);
        theta[i].resize(k*nn[i]);
        hyper[i].resize(kp*nn[i]);
        mb[i-indx].t.treetovec(&id[i][0],&v[i][0],&c[i][0],&theta[i][0],&hyper[i][0],k,kp);
    }
}


//--------------------------------------------------
//Local load tree
void amxbrt::local_loadtree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta){
  size_t indx=iter*m;
  for(size_t i=(indx+(size_t)beg);i<(indx+(size_t)end);i++)
    mb[i-indx].t.vectotree(nn[i],&id[i][0],&v[i][0],&c[i][0],&theta[i][0],k);
}

//--------------------------------------------------
//pr for brt
void amxbrt::pr_vec()
{
   std::cout << "***** ambrt object:\n";
   cout << "Number of trees in product representation:" << endl;
   cout << "        m:   m=" << m << endl;
   cout << "Conditioning info on each individual tree:" << endl;
   cout << "   mean:   tau=" << ci.tau << endl;
   cout << "   mean:   beta0 =" << ci.beta0 << endl;
   if(!ci.sigma)
     cout << "         sigma=[]" << endl;
   else
     cout << "         sigma=[" << ci.sigma[0] << ",...," << ci.sigma[di->n-1] << "]" << endl;
   brt::pr_vec();
   cout << "**************Trees in sum representation*************:" << endl;
   for(size_t j=0;j<m;j++) mb[j].t.pr_vec();
}

//--------------------------------------------------
//Collapse BART ensemble into one supertree
//The single supertree created will be stored in st.
void amxbrt::collapseensemble()
{
   tree::npv bots;
   resetst();

   for(size_t j=1;j<m;j++)
   {
      st.getbots(bots);
      //collapse each tree j=1..m into the supertree
      for(size_t i=0;i<bots.size();i++)
         collapsetree_vec(st,bots[i],this->gettree(j)); //mb[j]->t);
      bots.clear();
   }

}


//--------------------------------------------------
// rpath related methods
//--------------------------------------------------
void amxbrt::rpath_adapt(){
  for(size_t j=0;j<m;j++) {
    mb[j].rpath_adapt();
  }
}

void amxbrt::drawgamma(rn &gen){
    for(size_t j=0;j<m;j++) {
        //Draw parameter vector in the jth tree
        mb[j].drawgamma(gen);
    }
} 


void amxbrt::drawgamma_mpi(rn &gen){
    for(size_t j=0;j<m;j++) {
        //Draw parameter vector in the jth tree
        mb[j].drawgamma_mpi(gen);
    }
}


// TODO: Inefficient with the bnv
void amxbrt::local_predict_vec_rpath(diterator& diter, finfo& fipred){
    tree::npv bnv;
    vxd thetavec_temp(k); 
    vxd phix;
    std::map<tree::tree_p,tree::npv> pathmap;
    std::map<tree::tree_p,double> lbmap, ubmap;
    std::map<tree::tree_p,int> Umap, Lmap, vmap;

    // Get bots and then get the path and bounds
    for(size_t j=0;j<m;j++){
        //bnv.clear();
        //mb[j].t.getbots(bnv);
        mb[j].t.rgitree(*xi);
        //get_phix_bounds(lbmap, ubmap, Lmap, Umap, vmap);
        //get_phix_bounds(bnv, lbmap, ubmap, pathmap);
    }
      
    for(;diter<diter.until();diter++){
        thetavec_temp = vxd::Zero(k);   
        double *xx = diter.getxp(); 

        for(size_t j=0;j<m;j++){
            std::map<tree::tree_p,double> phixmap;
            std::map<tree::tree_p,double> logpathprob;
            logpathprob[mb[j].t.getptr(t.nid())] = 0; // init at root

            // Newer
            mb[j].t.calcphix(xx,*xi,phixmap,logpathprob,mb[j].get_gamma(),rpi.q); 
            //std::map<tree::tree_p,double>::iterator pit;
            for(auto i = phixmap.begin(); i != phixmap.end();i++){
                thetavec_temp = thetavec_temp + ((i->first)->getthetavec())*(i->second);   
            }

            /*
            // Get Bots
            bnv.clear();
            mb[j].t.getbots(bnv);
            // Reset phix
            phix = vxd::Ones(bnv.size());
            mb[j].get_phix(xx,phix,bnv,lbmap,ubmap,pathmap);
            for(size_t l=0;l<bnv.size();l++){
                //if(std::isnan(tempphix)){ cout << "nan phix ..." << endl; }
                thetavec_temp = thetavec_temp + (bnv[l]->getthetavec())*phix(l);
            }
            */
            
        }
        diter.sety(fipred.row(*diter)*thetavec_temp);
    }
}


// TODO: Inefficient with the bnv
void amxbrt::local_predict_thetavec_rpath(diterator& diter, mxd& wts){
    tree::npv bnv;
    vxd thetavec_temp(k); 
    vxd phix;
    std::map<tree::tree_p,tree::npv> pathmap;
    std::map<tree::tree_p,double> lbmap, ubmap;
    std::map<tree::tree_p,int> Umap, Lmap, vmap;


    // Get bots and then get the path and bounds
    for(size_t j=0;j<m;j++){
        mb[j].t.rgitree(*xi);
        //bnv.clear();
        //mb[j].t.getbots(bnv);
        //get_phix_bounds(lbmap, ubmap, Lmap, Umap, vmap);
        //get_phix_bounds(bnv, lbmap, ubmap, pathmap);
    }
      
    for(;diter<diter.until();diter++){
        thetavec_temp = vxd::Zero(k);   
        double *xx = diter.getxp(); 
        for(size_t j=0;j<m;j++){
            std::map<tree::tree_p,double> phixmap;
            std::map<tree::tree_p,double> logpathprob;
            logpathprob[mb[j].t.getptr(t.nid())] = 0; // init at root

            // Newer
            mb[j].t.calcphix(xx,*xi,phixmap,logpathprob,mb[j].get_gamma(),rpi.q); 
            for(auto i = phixmap.begin(); i != phixmap.end();i++){
                thetavec_temp = thetavec_temp + ((i->first)->getthetavec())*(i->second);   
            }

        }
        /*
        for(size_t j=0;j<m;j++){
            // Get bots
            bnv.clear();
            mb[j].t.getbots(bnv);
            // Reset phix
            phix = vxd::Ones(bnv.size());
            mb[j].get_phix(xx,phix,bnv,lbmap,ubmap,pathmap);
            for(size_t l=0;l<bnv.size();l++){
                //if(std::isnan(tempphix)){ cout << "nan phix ..." << endl; }
                thetavec_temp = thetavec_temp + (bnv[l]->getthetavec())*phix(l);
            }
        }
        */
        wts.col(*diter) = thetavec_temp;
    }
}


// Getter for the gamma parameter
std::vector<double> amxbrt::getgamma(){
    std::vector<double> outgamma;
    for(size_t j=0;j<m;j++){
        outgamma.push_back(mb[j].get_gamma());
    }
    return(outgamma);
}


// Sample tree prior -- used for variogram
void amxbrt::sample_tree_prior(rn& gen){
    for(size_t j=0;j<m;j++){
        mb[j].sample_tree_prior(gen);
    }
}


// Get phi matrix for each tree
void amxbrt::get_phix_list(diterator &diter, std::vector<mxd, Eigen::aligned_allocator<mxd>> &phix_list, size_t np){
    vxd phix;
    mxd phix_mat;
    tree::npv bnv; 
    //std::vector<mxd, Eigen::aligned_allocator<mxd>> phixlist(m); //An std vector of dim k -- each element is an nd X np eigen matrix
    std::map<tree::tree_p,double> lbmap;
    std::map<tree::tree_p,double> ubmap;
    std::map<tree::tree_p,tree::npv> pathmap;

    // Get bots and then get the path and bounds
    for(size_t j=0;j<m;j++){
        bnv.clear();
        mb[j].t.getbots(bnv);
        get_phix_bounds(bnv, lbmap, ubmap, pathmap);
        phix_mat = mxd::Ones(np,bnv.size())/bnv.size();
        phix_list.push_back(phix_mat);
    }
      
    for(;diter<diter.until();diter++){   
        double *xx = diter.getxp(); 
        for(size_t j=0;j<m;j++){
            // Get bots
            bnv.clear();
            mb[j].t.getbots(bnv);
            // Reset phix
            phix = vxd::Ones(bnv.size())/bnv.size(); // Reset phix 
            mb[j].get_phix(xx,phix,bnv,lbmap,ubmap,pathmap); // Get phix for this x and tree
            phix_list[j].row(*diter) = phix; // Edit the corresponding row in the jth matrix
        }        
    }
}