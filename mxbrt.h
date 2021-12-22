#ifndef GUARD_mxbrt_h
#define GUARD_mxbrt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"

//Include the Eigen header files
#include "Eigen/Dense"

//Define the mxinfo class -- inherits from sinfo
class mxsinfo : public sinfo{
    public:
        //Constructors:
        mxsinfo():sinfo(),k(1),sumfft(Eigen::MatrixXd::Zero(k,k)), sumfr(Eigen::VectorXd::Zero(k)) {} //Initialize mxinfo with default settings
        mxsinfo(sinfo& is, int k0, Eigen::MatrixXd sff, Eigen::VectorXd sfr):sinfo(is), k(k0), sumfft(sff), sumfr(sfr) {} //Construct mxinfo instance with values -- need to use references
        mxsinfo(const mxsinfo& is):sinfo(is),k(is.k),sumfft(is.sumfft),sumfr(is.sumfr){} //Input object of mxinfro (is) and instantiate new mxinfo 

        virtual ~mxsinfo() {} //free memory -- need to figure out how this works -- can't use the first constructor in int main 

        //Initialize sufficient stats
        Eigen::MatrixXd sumfft; //Computes F^t*F by summing over fi*fi^t for the observations in each tnode (fi = vector of dimension K)
        Eigen::VectorXd sumfr; //Computes F^t*R by summing over fi*ri for observations in each tnode (fi = vector and ri = scalar) 
        int k; //number of models, so equal to number of columns in sumfft. This is needed in order to initialize Zero matrix/vector in eigen

        //Define Operators -- override from sinfo class
        //Compound addition operator - used when adding sufficient statistics
        virtual sinfo& operator+=(const sinfo& rhs){
            sinfo::operator+=(rhs); 
            const mxsinfo& mrhs = static_cast<const mxsinfo&>(rhs); //Cast rhs as an mxinfo instance.  
            sumfft += mrhs.sumfft;
            sumfr += mrhs.sumfr;
            return *this; //returning *this should indicate that we return updated sumfft and sumfr while also using a pointer
        }

        //Compound assignment operator for sufficient statistics
        virtual sinfo& operator=(const sinfo& rhs){
            if(&rhs != this){
                sinfo::operator=(rhs); //--Figure out what this line does
                const mxsinfo& mrhs=static_cast<const mxsinfo&>(rhs);
                this->sumfft = mrhs.sumfft; 
                this->sumfr = mrhs.sumfr;
                this->k = mrhs.k; //May not need this assignment
                return *this; 
            }
            return *this; //returning *this should indicate that we return updated sumfft and sumfr while also using a pointer
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
            std::cout << "sumfft = \n" << sumfft << std::endl;
            std::cout << "sumfr = \n" << sumfr << std::endl;
            std::cout << "K = " << k << std::endl;
        }
};


class mxbrt : public brt{
public:
   //--------------------
   //classes
   // tprior and mcmcinfo are same as in brt
   class cinfo { //parameters for end node model prior
   public:
      cinfo():tau(1.0),sigma(0) {}
      double tau;
      double* sigma;
   };
   //--------------------
   //constructors/destructors
   mxbrt():brt() {}
   //--------------------
   //methods
   void drawvec(rn& gen);
   void drawvec_mpislave(rn& gen);
   void setci(double tau, double* sigma) { ci.tau=tau; ci.sigma=sigma; }
   virtual vxd drawnodethetavec(sinfo& si, rn& gen);
   virtual double lm(sinfo& si);
   virtual void add_observation_to_suff(diterator& diter, sinfo& si);
   virtual sinfo* newsinfo() { return new mxsinfo; }
   virtual std::vector<sinfo*>& newsinfovec() { std::vector<sinfo*>* si= new std::vector<sinfo*>; return *si; }
   virtual std::vector<sinfo*>& newsinfovec(size_t dim) { std::vector<sinfo*>* si = new std::vector<sinfo*>; si->resize(dim); for(size_t i=0;i<dim;i++) si->push_back(new mxsinfo); return *si; }
   virtual void local_mpi_reduce_allsuff(std::vector<sinfo*>& siv);
   virtual void local_mpi_sr_suffs(sinfo& sil, sinfo& sir);
   void pr_vec();

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

