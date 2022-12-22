#ifndef GUARD_amcbrt_h
#define GUARD_amcbrt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "mcbrt.h"

class amcbrt : public mcbrt {
public:
   //--------------------
   //classes
   // tprior and mcmcinfo are same as in brt
   // cinfo same as in mxbrt
   //--------------------
   //constructors/destructors
   amcbrt(): mcbrt(),st(0),m(200),mb(m),notjmus(m),divec(m) {}
   amcbrt(size_t im): mcbrt(),st(0),m(im),mb(m),notjmus(m),divec(m) {}
   //amxbrt(size_t im, size_t ik): mxbrt(ik),st(0),m(im),mb(m),notjmus(m),divec(m) {}
   virtual ~amcbrt() {
      if(!notjmus.empty()) {
         for(size_t j=0;j<m;j++) notjmus[j].clear();
         notjmus.clear();
         for(size_t j=0;j<m;j++) delete divec[j];
      }
      st.tonull();
   }

   //--------------------
   //methods
   void drawvec(rn& gen);
   void drawvec_mpislave(rn& gen);
   void adapt();
   void setmpirank(int rank) { this->rank = rank; for(size_t j=0;j<m;j++) mb[j].setmpirank(rank); }  //only needed for MPI
   void setmpicvrange(int* lwr, int* upr) { this->chv_lwr=lwr; this->chv_upr=upr; for(size_t j=0;j<m;j++) mb[j].setmpicvrange(lwr,upr); } //only needed for MPI
   void setci(double mu1, double mu2, double tau1, double tau2, double* sigma) { ci.tau1=tau1;ci.tau2=tau2;ci.mu1=mu1;ci.mu2=mu2;ci.sigma=sigma;
                for(size_t j=0;j<m;j++) mb[j].setci(tau1,tau2,mu1,mu2,sigma); }
   void settc(int tc) { this->tc = tc; for(size_t j=0;j<m;j++) mb[j].settc(tc); }
   void setxi(xinfo *xi) { this->xi=xi; for(size_t j=0;j<m;j++) mb[j].setxi(xi); }
   void setfi(finfo *fi, size_t k) {this->fi = fi; this->k = k; this-> nsprior = false ;for(size_t j=0;j<m;j++) mb[j].setfi(fi,k); }
   void setdata_vec(dinfo *di);
   void settp(double alpha, double beta) { tp.alpha=alpha;tp.beta=beta; for(size_t j=0;j<m;j++) mb[j].settp(alpha,beta); }
   tree::tree_p gettree(size_t i) { return &mb[i].t; }
   void setmi(double pbd, double pb, size_t minperbot, bool dopert, double pertalpha, double pchgv, std::vector<std::vector<double> >* chgv)
             { mi.pbd=pbd; mi.pb=pb; mi.minperbot=minperbot; mi.dopert=dopert;
               mi.pertalpha=pertalpha; mi.pchgv=pchgv; mi.corv=chgv; 
               for(size_t j=0;j<m;j++) mb[j].setmi(pbd,pb,minperbot,dopert,pertalpha,pchgv,chgv); }
   void setstats(bool dostats) { mi.dostats=dostats; for(size_t j=0;j<m;j++) mb[j].setstats(dostats); if(dostats) mi.varcount=new unsigned int[xi->size()]; }
   void pr_vec();
   // drawnodetheta, lm, add_observation_to_suff and newsinfo/newsinfovec unused here.

   // convert BART ensemble to single supertree
   void resetst() { st.tonull(); st=mb[0].t; } //copy mb0's tree to st.
   void collapseensemble();

    //--------------------
    //data
    //--------------------------------------------------
    //stuff that maybe should be protected
    tree st;

protected:
    //--------------------
    //model information
    size_t m;  //number of trees in sum representation
    std::vector<mcbrt> mb;  // the vector of individual mu trees for sum representation
    //--------------------
    //data
    std::vector<std::vector<double> > notjmus;
    std::vector<dinfo*> divec;
    //--------------------
    //mcmc info
    //--------------------
    //methods
    virtual void local_setf_vec(diterator& diter);  //set the vector of predicted values
    virtual void local_setr_vec(diterator& diter);  //set the vector of residuals
    virtual void local_predict_vec(diterator& diter, finfo& fipred); // predict y at the (npred x p) settings *di.x
    virtual void local_predict_thetavec(diterator& diter, mxd& wts); // extract vector parameters at each *di.x settings
    virtual void local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                    std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
    virtual void local_loadtree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                    std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);

};

#endif