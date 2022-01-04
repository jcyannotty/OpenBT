#ifndef GUARD_mxbrt_h
#define GUARD_mxbrt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"

//Include the Eigen header files
#include "Eigen/Dense"

//typedef defined in tree class --- mxd = Eigen::MatrixXd and vxd = Eigen::VectorXd

//Define the mxinfo class -- inherits from sinfo
class mxsinfo : public sinfo{
    public:
        //Constructors:
        mxsinfo():sinfo(),k(2),sumffw(mxd::Zero(2,2)), sumfyw(vxd::Zero(2)), sumyyw(0.0) {} //Initialize mxinfo with default settings
        mxsinfo(sinfo& is, int k0, mxd sff, vxd sfy, double syy):sinfo(is), k(k0), sumffw(sff), sumfyw(sfy), sumyyw(syy) {} //Construct mxinfo instance with values -- need to use references
        mxsinfo(const mxsinfo& is):sinfo(is),k(is.k),sumffw(is.sumffw),sumfyw(is.sumfyw),sumyyw(is.sumyyw){} //Input object of mxinfro (is) and instantiate new mxinfo 

        virtual ~mxsinfo() {} 

        //Initialize sufficient stats
        Eigen::MatrixXd sumffw; //Computes F^t*F/sig2 by summing over fi*fi^t for the observations in each tnode (fi = vector of dimension K)
        Eigen::VectorXd sumfyw; //Computes F^t*Y/sig2 by summing over fi*yi for observations in each tnode (fi = vector and yi = scalar) 
        double sumyyw; //computes Y^t*Y/sig2 by summing over yi*yi for observations in each tnode
        int k; //number of models, so equal to number of columns in sumffw. This is needed in order to initialize Zero matrix/vector in eigen

        //Define Operators -- override from sinfo class
        //Compound addition operator - used when adding sufficient statistics
        virtual sinfo& operator+=(const sinfo& rhs){
            sinfo::operator+=(rhs); 
            const mxsinfo& mrhs = static_cast<const mxsinfo&>(rhs); //Cast rhs as an mxinfo instance.  
            sumffw += mrhs.sumffw;
            sumfyw += mrhs.sumfyw;
            sumyyw += mrhs.sumyyw;
            return *this; //returning *this should indicate that we return updated sumffw and sumfyw while also using a pointer
        }

        //Compound assignment operator for sufficient statistics
        virtual sinfo& operator=(const sinfo& rhs){
            if(&rhs != this){
                sinfo::operator=(rhs); //--Figure out what this line does
                const mxsinfo& mrhs=static_cast<const mxsinfo&>(rhs);
                this->sumffw = mrhs.sumffw; 
                this->sumfyw = mrhs.sumfyw;
                this->sumyyw = mrhs.sumyyw;
                this->k = mrhs.k; //May not need this assignment
                return *this; 
            }
            return *this; //returning *this should indicate that we return updated sumffw and sumfr while also using a pointer
        }

        //Addition operator -- defined in terms of the compund operator above. Use for addition across two instances of mxinfo
        const mxsinfo operator+(const mxsinfo& other) const{
            mxsinfo result = *this;
            result += other;
            return result;
        }

        //Print mxinfo instance
        void print_mx(){
            std::cout << "**************************************" << std::endl; 
            std::cout << "Model mixing sufficient statistics for this terminal node" << std::endl;
            std::cout << "sumffw = \n" << sumffw << std::endl;
            std::cout << "\nsumfyw = \n" << sumfyw << std::endl;
            std::cout << "\n sumyyw = " << sumyyw << std::endl;
            std::cout << "k = " << k << std::endl;
            std::cout << "n = " << n << std::endl;
            std::cout << "**************************************" << std::endl;
        }
};


class mxbrt : public brt{
public:
    //--------------------
    //classes
    // tprior and mcmcinfo are same as in brt
        class cinfo{
        public:
            cinfo():beta0(1.0), tau(1.0), sigma(0), nu(1.0), lambda(1.0) {} //beta0 = scalar in the prior mean vector, tau = prior stdev for tnode parameters, sigma = stdev of error 
            double beta0, tau;
            double* sigma; //use pointer since this will be changed as mcmc iterates
            double nu, lambda;
        };
    //--------------------
    //constructors/destructors
    mxbrt():brt() {}
    //--------------------
    //methods
    void drawvec(rn& gen);
    void drawvec_mpislave(rn& gen);
    void setci(double tau, double beta0, double* sigma) { ci.tau=tau; ci.beta0 = beta0; ci.sigma=sigma; } 
    virtual vxd drawnodethetavec(sinfo& si, rn& gen);
    virtual double lm(sinfo& si);
    virtual void add_observation_to_suff(diterator& diter, sinfo& si);
    virtual sinfo* newsinfo() { return new mxsinfo; }
    virtual std::vector<sinfo*>& newsinfovec() { std::vector<sinfo*>* si= new std::vector<sinfo*>; return *si; }
    virtual std::vector<sinfo*>& newsinfovec(size_t dim) { std::vector<sinfo*>* si = new std::vector<sinfo*>; si->resize(dim); for(size_t i=0;i<dim;i++) si->push_back(new mxsinfo); return *si; }
    virtual void local_mpi_reduce_allsuff(std::vector<sinfo*>& siv);
    virtual void local_mpi_sr_suffs(sinfo& sil, sinfo& sir);
    void pr_vec();

    //Method for sampling homoscedastic variance for paramter sigma^2 -- not sure if this works
    void setvi(double nu, double lambda) {ci.nu = nu; ci.lambda = lambda; } //Use to change the defualt parameters  
    void drawsigma(rn& gen); //Gibbs Sampler
    double getsigma() { return ci.sigma[0];}

    //Consider adding covariance inversion to local_subsuff-----******

    //--------------------
    //data
    //--------------------------------------------------
    //stuff that maybe should be protected
    protected:
    //--------------------
    //model information
    cinfo ci; //conditioning info (e.g. other parameters and prior and end node models)
    //--------------------
    //data
    //--------------------
    //mcmc info
    //--------------------
    //methods
};

#endif

