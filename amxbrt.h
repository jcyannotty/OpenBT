#ifndef GUARD_amxbrt_h
#define GUARD_amxbrt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "mxbrt.h"


class amxbrt : public mxbrt {
public:
   //--------------------
   //classes
   // tprior and mcmcinfo are same as in brt
   // cinfo same as in mxbrt
   //--------------------
   //constructors/destructors
   amxbrt(): mxbrt(),st(0),m(200),mb(m),notjmus(m),divec(m) {}
   amxbrt(size_t im): mxbrt(),st(0),m(im),mb(m),notjmus(m),divec(m) {}
   virtual ~amxbrt() {
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
   void setci(double tau, double beta0, double* sigma) { ci.tau=tau; ci.sigma=sigma; ci.beta0=beta0; for(size_t j=0;j<m;j++) mb[j].setci(tau,beta0,sigma); }
   void settc(int tc) { this->tc = tc; for(size_t j=0;j<m;j++) mb[j].settc(tc); }
   void setxi(xinfo *xi) { this->xi=xi; for(size_t j=0;j<m;j++) mb[j].setxi(xi); }
   void setfi(finfo *fi, int k) {this->fi = fi; this->k = k; for(size_t j=0;j<m;j++) mb[j].setfi(fi,k); }
   void setdata_mix(dinfo *di);
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

    /*
   // function for calculating Sobol-based variable activity indices
   void sobol(std::vector<double>& Si, std::vector<double>&Sij, std::vector<double>& TSi, double& V, std::vector<double>& minx, std::vector<double>& maxx, size_t p);

   // function for converting an ensemble to vector hyperrectangle format, needed for Pareto Front multiobjective optimization (see mopareto.cpp)
   void ens2rects(std::vector<std::vector<double> >& asol, std::vector<std::vector<double> >& bsol, 
                  std::vector<double>& thetasol, std::vector<double>& minx,
                  std::vector<double>& maxx, size_t p);
    */
    
    //--------------------
    //data
    //--------------------------------------------------
    //stuff that maybe should be protected
    tree st;

protected:
    //--------------------
    //model information
    size_t m;  //number of trees in sum representation
    std::vector<mxbrt> mb;  // the vector of individual mu trees for sum representation
    //--------------------
    //data
    std::vector<std::vector<double> > notjmus;
    std::vector<dinfo*> divec;
    //--------------------
    //mcmc info
    //--------------------
    //methods
    virtual void local_setf_mix(diterator& diter);  //set the vector of predicted values
    virtual void local_setr_mix(diterator& diter);  //set the vector of residuals
    virtual void local_predict_mix(diterator& diter, finfo& fipred); // predict y at the (npred x p) settings *di.x
    virtual void local_savetree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                    std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);
    virtual void local_loadtree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                    std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta);

};


#endif