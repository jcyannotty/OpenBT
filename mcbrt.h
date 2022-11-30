#ifndef GUARD_mcbrt_h
#define GUARD_mcbrt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"
#include "brtfuns.h"

//Include the Eigen header files
#include "Eigen/Dense"

//Define the mxinfo class -- inherits from sinfo
class mcinfo : public sinfo{
    public:
        //Notation:
        /* y = field data, z = model runs, w = data precision (wf or wc for field/computer data when needed to differentiate), 
           nf = number of field obs, n = total number of data (field obs and simulator runs)
        */ 
        
        //Constructors:
        mcinfo():sinfo(),sumyw(0.0), sumzw(0.0), sumwf(0.0), sumwc(0.0), nf(0), subtree_info(false){} //Initialize mxinfo with default settings
        mcinfo(const mcinfo& is):sinfo(is),sumyw(is.sumyw),sumzw(is.sumzw),sumwf(is.sumwf),sumwc(is.sumwc),nf(is.nf), subtree_info(false) {}
        /*
        mcinfo(size_t ik):sinfo(),sumffw(mxd::Zero(2,2)), sumfyw(vxd::Zero(ik)), sumyyw(0.0), sump(vxd::Zero(ik)) {} //Initialize mxinfo with number of columns in fi matrix
        mcinfo(sinfo& is, int ik, mxd sff, vxd sfy, double syy):sinfo(is), k(ik), sumffw(sff), sumfyw(sfy), sumyyw(syy), sump(vxd::Zero(ik)) {} //Construct mxinfo instance with values -- need to use references 
        mcinfo(sinfo& is, int ik, mxd sff, vxd sfy, double syy, vxd sp, vxd sp2):sinfo(is), k(ik), sumffw(sff), sumfyw(sfy), sumyyw(syy), sump(sp) {} //Construct mxinfo instance with values with input for discrepancy -- need to use references
        mcinfo(const mcinfo& is):sinfo(is),k(is.k),sumffw(is.sumffw),sumfyw(is.sumfyw),sumyyw(is.sumyyw), sump(is.sump) {} //Input object of mxinfro (is) and instantiate new mxinfo 
        */
        
        virtual ~mcinfo() {} 

        //Initialize sufficient stats
        double sumyw; // Sum of field data weighted by precision
        double sumzw; // Sum of model runs wieghted by precision
        double sumwf; // Sum of field data precision
        double sumwc; // Sum of model run precision  
        size_t nf; // number of field obs 
        
        // Sufficient stats that are collected for a subtree 
        // These are NEVER passed through the MPI, rather they are computed after all data has been passed
        // Important for the proposals and drawing theta
        std::vector<double> subtree_var; // Variances for the common nodes in the subtree (those that are not involved in birth/death) 
        std::vector<double> subtree_mean; // Means for the common nodes in the subtree (those that are not involved in birth/death)
        bool subtree_info; // Does this instance of mcinfo contain information for the rest of the subtree (other than its individual node info)?

        // Setters for the subtree info
        void setsubtreeinfo(std::vector<mcinfo*> mcv, double mu1, double tau1){            
            size_t nc = 0;
            double t1_sqr = tau1*tau1;
            double mtilde, vtilde, vhat, rhat;
            for(int i=0;i<mcv.size();i++){
                // Compute the mean and variance of theta1 given model runs
                vtilde = 1/((*mcv[i]).sumwc + 1/t1_sqr);
                mtilde = vtilde*((*mcv[i]).sumyw + mu1/t1_sqr);
                
                // Compute the required field data information 
                rhat = (*mcv[i]).sumyw/(*mcv[i]).sumwf;
                vhat = 1/(*mcv[i]).sumwf;

                // Pushback into the subtree vectors
                subtree_var.push_back(vhat + vtilde);
                subtree_mean.push_back(rhat - mtilde);
                subtree_info = true;
            }
        }

        //Define Operators -- override from sinfo class
        //Compound addition operator - used when adding sufficient statistics
        virtual sinfo& operator+=(const sinfo& rhs){
            // Standard overload operations
            sinfo::operator+=(rhs); 
            const mcinfo& mrhs = static_cast<const mcinfo&>(rhs); //Cast rhs as an mxinfo instance.  
            sumyw += mrhs.sumyw;
            sumzw += mrhs.sumzw;
            sumwf += mrhs.sumwf;
            sumwc += mrhs.sumwc;
            nf += mrhs.nf;

            // Overloading addition operator for the subtree vectors IF mrhs.subtree_info = true
            if(mrhs.subtree_info){
                // mrhs has subtree_info -- add it to this instance of mcinfo
                subtree_info = true;
                // using push_back here just incase this mcinfo instance already has subtree info and this mrhs is just adding more
                for(int i=0;i<mrhs.subtree_mean.size();i++){
                    subtree_mean.push_back(mrhs.subtree_mean[i]);
                    subtree_var.push_back(mrhs.subtree_var[i]);
                }
            }
            return *this; //returning *this should indicate that we return updated sumffw and sumfyw while also using a pointer
        }

        //Compound assignment operator for sufficient statistics
        virtual sinfo& operator=(const sinfo& rhs){
            if(&rhs != this){
                sinfo::operator=(rhs); 
                const mcinfo& mrhs=static_cast<const mcinfo&>(rhs);
                this->sumyw = mrhs.sumyw; 
                this->sumzw = mrhs.sumzw;
                this->sumwf = mrhs.sumwf;
                this->sumwc = mrhs.sumwc;
                this->nf = mrhs.nf;

                // Overlod for subtree info when applicable
                if(mrhs.subtree_info){
                    this->subtree_mean = mrhs.subtree_mean;
                    this->subtree_var = mrhs.subtree_var;
                }
                return *this; 
            }
            return *this; //returning *this should indicate that we return updated sumffw and sumfr while also using a pointer
        }

        //Addition operator -- defined in terms of the compund operator above. Use for addition across two instances of mxinfo
        const mcinfo operator+(const mcinfo& other) const{
            mcinfo result = *this;
            result += other;
            return result;
        }

        //Print mxinfo instance
        void print(){
            std::cout << "**************************************" << std::endl; 
            std::cout << "Model calibration sufficient statistics for this terminal node" << std::endl;
            std::cout << "sumyw = " << sumyw << std::endl;
            std::cout << "sumzw = " << sumzw << std::endl;
            std::cout << "sumwf = " << sumwf << std::endl;
            std::cout << "sumwc = " << sumwc << std::endl;
            std::cout << "nf = " << nf << std::endl;
            std::cout << "nc = " << n-nf << std::endl;
            std::cout << "n = " << n << std::endl;
            std::cout << "**************************************" << std::endl;
        }
};


class mcbrt : public brt{
public:
    //--------------------
    //classes
    // tprior and mcmcinfo are same as in brt
        class cinfo{
        public:
            cinfo():mu1(0.0), mu2(1.0), tau1(1.0), tau2(1.0), sigma(0), nu(1.0), lambda(1.0) {}  
            double mu1, mu2, tau1, tau2; //mu1 & tau1 = prior mean/std for eta, mu2 & tau2 = prior mean/std for delta 
            double* sigma; 
            double nu, lambda;
        };
    //--------------------
    //constructors/destructors
    mcbrt():brt() {}
    //mxbrt(size_t ik):brt(ik) {}
    //--------------------
    //methods
    void drawvec(rn& gen);
    void drawvec_mpislave(rn& gen);
    void setci(double mu1,double mu2, double tau1, double tau2, double* sigma) {ci.mu1=mu1; ci.tau1=tau1; ci.mu2=mu2; ci.tau2=tau2; ci.sigma=sigma; }
    virtual vxd drawnodethetavec(sinfo& si, rn& gen);
    virtual double lm(sinfo& si);
    virtual void add_observation_to_suff(diterator& diter, sinfo& si);
    virtual sinfo* newsinfo() { return new mcinfo(); }
    virtual std::vector<sinfo*>& newsinfovec() { std::vector<sinfo*>* si= new std::vector<sinfo*>; return *si; }
    virtual std::vector<sinfo*>& newsinfovec(size_t dim) { std::vector<sinfo*>* si = new std::vector<sinfo*>; si->resize(dim); for(size_t i=0;i<dim;i++) si->push_back(new mcinfo()); return *si; }
    virtual void local_mpi_reduce_allsuff(std::vector<sinfo*>& siv);
    virtual void local_mpi_sr_suffs(sinfo& sil, sinfo& sir);
    void pr_vec();

    // Methods which are not overridden in other inherited classes (mbrt, mxbrt) - but override is required here
    void allsuff(tree::npv& bnv,std::vector<sinfo*>& siv);  //assumes brt.t is the root node
    void local_subsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv); //does NOT assume brt.t is the root node.
    void local_allsuff(diterator& diter, tree::npv& bnv,std::vector<sinfo*>& siv);
    
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
    void local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir); //assumes brt.t is the root node
    void local_getsuff(diterator& diter, tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir); //assumes brt.t is the root node
};

#endif