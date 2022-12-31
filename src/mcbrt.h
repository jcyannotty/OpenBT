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
           --- Might not need nf after all??
        */ 
        
        //Constructors:
        mcinfo():sinfo(),sumyw(0.0),sumzw(0.0),sumwf(0.0),sumwc(0.0),nf(0),subtree_info(false),sibling_info(false),subtree_node(false) {} //Initialize mxinfo with default settings
        mcinfo(bool st):sinfo(),sumyw(0.0),sumzw(0.0),sumwf(0.0),sumwc(0.0),nf(0),subtree_info(false),sibling_info(false),subtree_node(st) {}
        mcinfo(double syw, double szw, double swf, double swc, size_t nf):sinfo(),sumyw(syw),sumzw(szw),sumwf(swf),sumwc(swc),nf(nf),subtree_info(false),sibling_info(false),subtree_node(false) {}
        mcinfo(const mcinfo& is):sinfo(is),sumyw(is.sumyw),sumzw(is.sumzw),sumwf(is.sumwf),sumwc(is.sumwc),nf(is.nf),subtree_info(false), sibling_info(false),subtree_node(false) {}
        
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
        double sibling_var; // Variance for the sibling of a node involved in birth/death -- essential in birth/death
        double sibling_mean; // Mean for the sibling of a node involved in birth/death -- essential in birth/death
        bool sibling_info; // Does this instance of mcinfo contain information of a sibling node
        bool subtree_node; // Is this instance of mcinfo in a subtree -- needed to differentiate between cases in lm and drawtheta

        std::vector<double> subtree_sumyw;
        std::vector<double> subtree_sumzw;
        std::vector<double> subtree_sumwf;
        std::vector<double> subtree_sumwc;
        double sibling_sumyw;
        double sibling_sumzw;
        double sibling_sumwf;
        double sibling_sumwc;

        // Computing mean and variance for calculation in the joint model
        std::vector<double> getmoments(double mu1, double tau1){
            std::vector<double> out_meanvar;
            double t1_sqr = tau1*tau1;
            double mtilde = 0, vtilde = 0, vhat = 0, rhat = 0;
            
            if(sumwf>0){
                // Compute the modularized mean
                vtilde = 1/(sumwc + 1/t1_sqr);
                mtilde = vtilde*(sumzw + mu1/t1_sqr);
                
                // Compute the required field data information 
                rhat = sumyw/sumwf;
                vhat = 1/sumwf;
            }
            
            // Compute the mean and variance info
            out_meanvar.push_back(rhat - mtilde);
            out_meanvar.push_back(vhat + vtilde);
            return out_meanvar;            
        }

        // Apply getmoments to subtree information
        void setsubtreemoments(double mu1, double tau1){
            double node_mean, node_var;
            std::vector<double> node_vec;
            for(size_t i=0;i<subtree_sumwc.size();i++){
                // Constructor to create a new instance of mcinfo that allows us to use the getmoments() method 
                // set nf = 1 by default (nf value is not needed here)
                mcinfo mci(subtree_sumyw[i],subtree_sumzw[i],subtree_sumwf[i],subtree_sumwc[i],1);
                // Get the node mean and variance
                node_vec = mci.getmoments(mu1, tau1);
                node_mean = node_vec[0];
                node_var = node_vec[1];
                
                // Pushback into the subtree vectors
                subtree_mean.push_back(node_mean);
                subtree_var.push_back(node_var);
            }

            // Now set sibling info if this node has it as well and do the same process
            if(sibling_info){
                mcinfo mci(sibling_sumyw,sibling_sumzw,sibling_sumwf,sibling_sumwc,1); // set nf = 1 by default, its not important here
                node_vec = mci.getmoments(mu1, tau1);
                sibling_mean = node_vec[0];
                sibling_var = node_vec[1];            
            }
        }


        // Set subtree information -- this node will hold all information pooled across nodes in a calibration subtree
        void setsubtreeinfo(std::vector<mcinfo*> mcv){
            // Set subtree info to true
            subtree_info = true;
            for(size_t i=0;i<mcv.size();i++){
                // Store the required sufficient statistics
                subtree_sumyw.push_back((*mcv[i]).sumyw);
                subtree_sumzw.push_back((*mcv[i]).sumzw);
                subtree_sumwf.push_back((*mcv[i]).sumwf);
                subtree_sumwc.push_back((*mcv[i]).sumwc);
            }
        }

        // Resize subtree info vectors -- essential for the mpisubsuff on rank 0
        void resizesubtreeinfo(size_t sz){
            subtree_info = true;
            subtree_sumyw.resize(sz);
            subtree_sumzw.resize(sz);
            subtree_sumwf.resize(sz);
            subtree_sumwc.resize(sz);
        }

        // Set sibling information -- essential for birth and death steps. If the right child holds subtree_info,
        // then it must also hold the sibling info from the left child. Note, only one child needs to hold sibling and subtree info
        // Using pass by reference to match with existing treatment of suff stats within the local_getsuff code
        void setsiblinginfo(mcinfo &mci){
            // Set subtree info to true
            sibling_info = true;
            // Store the required sufficient statistics
            sibling_sumyw = mci.sumyw;
            sibling_sumzw = mci.sumzw;
            sibling_sumwf = mci.sumwf;
            sibling_sumwc = mci.sumwc;
        }

        /*
        // Setters for the subtree info
        void setsubtreeinfo(std::vector<mcinfo*> mcv, double mu1, double tau1){            
            double node_mean, node_var;
            std::vector<double> node_vec;
            for(int i=0;i<mcv.size();i++){
                // Get the node mean and variance
                node_vec = (*mcv[i]).getmoments(mu1, tau1);
                node_mean = node_vec[0];
                node_var = node_vec[1];
                
                // Pushback into the subtree vectors
                subtree_mean.push_back(node_mean);
                subtree_var.push_back(node_var);
                subtree_info = true;
            }
        }
        */

        /*
        void setsiblinginfo(mcinfo &mci, double mu1, double tau1){
            // Compute vtilde and mtilde for this node
            double t1_sqr = tau1*tau1;
            double mtilde, vtilde, vhat, rhat;

            vtilde = 1/(mci.sumwc + 1/t1_sqr);
            mtilde = vtilde*(mci.sumzw + mu1/t1_sqr);
            
            // Compute the required field data information 
            rhat = mci.sumyw/mci.sumwf;
            vhat = 1/mci.sumwf;

            // Pushback into the subtree vectors
            sibling_var = vhat + vtilde;
            sibling_mean = rhat - mtilde;
            sibling_info = true; 
        }
        */

        // Define Operators -- override from sinfo class
        // Compound addition operator - used when adding sufficient statistics
        // Note -- no need to overload operator for the sibling info!!
        virtual sinfo& operator+=(const sinfo& rhs){
            // Standard overload operations
            sinfo::operator+=(rhs); 
            const mcinfo& mrhs = static_cast<const mcinfo&>(rhs); //Cast rhs as an mxinfo instance.  
            sumyw += mrhs.sumyw;
            sumzw += mrhs.sumzw;
            sumwf += mrhs.sumwf;
            sumwc += mrhs.sumwc;
            nf += mrhs.nf;

            // Overload subtree_node propoerty -- may need to consider other cases
            if(mrhs.subtree_node && !subtree_node){
                subtree_node = true;
            }

            // Overloading addition operator for the subtree vectors IF mrhs.subtree_info = true
            if(mrhs.subtree_info && !subtree_info){
                // mrhs has subtree_info but this one does not!-- add it to this instance of mcinfo
                subtree_info = true;
                // using push_back here just incase this mcinfo instance already has subtree info and this mrhs is just adding more
                for(size_t i=0;i<mrhs.subtree_sumyw.size();i++){
                    //subtree_mean.push_back(mrhs.subtree_mean[i]);
                    //subtree_var.push_back(mrhs.subtree_var[i]);
                    subtree_sumyw.push_back(mrhs.subtree_sumyw[i]);
                    subtree_sumzw.push_back(mrhs.subtree_sumzw[i]);
                    subtree_sumwf.push_back(mrhs.subtree_sumwf[i]);
                    subtree_sumwc.push_back(mrhs.subtree_sumwc[i]);
                }
            }else if(mrhs.subtree_info && subtree_info){
                // mrhs has subtree_info and so does this one -- add the terms together to this instance of mcinfo
                for(size_t i=0;i<mrhs.subtree_sumyw.size();i++){
                    subtree_sumyw[i]+=mrhs.subtree_sumyw[i];
                    subtree_sumzw[i]+=mrhs.subtree_sumzw[i];
                    subtree_sumwf[i]+=mrhs.subtree_sumwf[i];
                    subtree_sumwc[i]+=mrhs.subtree_sumwc[i];
                }
            }
            // Overloading addition operator for sibling info -- adding two nodes that have sibling info (occurs in mpi only)
            if(mrhs.sibling_info && sibling_info){
                sibling_sumyw+=mrhs.sibling_sumyw;
                sibling_sumzw+=mrhs.sibling_sumzw;
                sibling_sumwf+=mrhs.sibling_sumwf;
                sibling_sumwc+=mrhs.sibling_sumwc;
            }
            return *this;
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
                    this->subtree_sumyw = mrhs.subtree_sumyw;
                    this->subtree_sumzw = mrhs.subtree_sumzw;
                    this->subtree_sumwf = mrhs.subtree_sumwf;
                    this->subtree_sumwc = mrhs.subtree_sumwc;
                    this->subtree_mean = mrhs.subtree_mean;
                    this->subtree_var = mrhs.subtree_var;
                    this->subtree_info = mrhs.subtree_info;
                }
                // Overlod for subtree info when applicable
                if(mrhs.sibling_info){
                    this->sibling_sumyw = mrhs.sibling_sumyw;
                    this->sibling_sumzw = mrhs.sibling_sumzw;
                    this->sibling_sumwf = mrhs.sibling_sumwf;
                    this->sibling_sumwc = mrhs.sibling_sumwc;
                    this->sibling_info = mrhs.sibling_info;
                }
                return *this; 
            }
            return *this;
        }

        //Addition operator -- defined in terms of the compund operator above. Use for addition across two instances of mxinfo
        const mcinfo operator+(const mcinfo& other) const{
            mcinfo result = *this;
            result += other;
            return result;
        }

        //Print mcinfo instance
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
            std::cout << "is subtree node = " << subtree_node << std::endl;
            std::cout << "has subtree info = " << subtree_info << std::endl;
            std::cout << "has sibling info = " << sibling_info << std::endl;
            if(subtree_info){
                std::cout << "subtree size = " << subtree_sumyw.size() << std::endl;
                std::cout << "subtree_sumyw = " << subtree_sumyw[0] << " ... " << subtree_sumyw[subtree_sumyw.size()-1] << std::endl;
                std::cout << "subtree_sumzw = " << subtree_sumzw[0] << " ... " << subtree_sumzw[subtree_sumyw.size()-1] << std::endl;
                std::cout << "subtree_sumwf = " << subtree_sumwf[0] << " ... " << subtree_sumwf[subtree_sumyw.size()-1] << std::endl;
                std::cout << "subtree_sumwc = " << subtree_sumwc[0] << " ... " << subtree_sumwc[subtree_sumyw.size()-1] << std::endl;
            }
            if(sibling_info){
                std::cout << "sibling_sumyw = " << sibling_sumyw << std::endl;
                std::cout << "sibling_sumzw = " << sibling_sumzw << std::endl;
                std::cout << "sibling_sumwf = " << sibling_sumwf << std::endl;
                std::cout << "sibling_sumwc = " << sibling_sumwc << std::endl;
            }

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
    void drawthetavec(rn& gen);
    virtual double drawtheta2(std::vector<sinfo*> sivec, rn& gen);
    virtual double drawtheta1(sinfo& si, rn& gen, double theta2);
    virtual vxd drawnodethetavec(sinfo& si, rn& gen);
    virtual double lm(sinfo& si);
    virtual void add_observation_to_suff(diterator& diter, sinfo& si);
    virtual sinfo* newsinfo() { return new mcinfo(); }
    virtual sinfo* newsinfo(bool st) { return new mcinfo(st); } // Initialize mcinfo for a calibration subtree node 
    virtual std::vector<sinfo*>& newsinfovec() { std::vector<sinfo*>* si= new std::vector<sinfo*>; return *si; }
    virtual std::vector<sinfo*>& newsinfovec(size_t dim) { std::vector<sinfo*>* si = new std::vector<sinfo*>; si->resize(dim); for(size_t i=0;i<dim;i++) si->push_back(new mcinfo()); return *si; }
    virtual void local_mpi_reduce_allsuff(std::vector<sinfo*>& siv);
    virtual void local_mpi_sr_suffs(sinfo& sil, sinfo& sir);
    void pr_vec();

    // Methods which are not overridden in other inherited classes (mbrt, mxbrt) - but override is required here
    void subsuff(tree::tree_p nx, tree::npv& bnv, std::vector<sinfo*>& siv);
    //void local_subsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv); //does NOT assume brt.t is the root node.
    void local_subsuff_nodecases(tree::tree_p nx, tree::tree_p subtree, tree::npv& bnv, std::vector<sinfo*>& siv);
    void local_subsuff_setroot(tree::tree_p nx,tree::tree_p &subtree,tree::tree_p &troot ,tree::npv &uroots);
    void local_subsuff_subtree(std::vector<sinfo*>& siv);
    void local_subsuff_subtree(tree::npv nxuroots, tree::tree_p nx, tree::npv& bnv, std::vector<sinfo*>& siv);
    //void local_mpisubsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv);
    
    // Methods which are nested within the above methods. Used to deal with various cases and keep code concise
    double lmnode(mcinfo &mci); // Node outside of subtree -- Used when doing proposals outside of a subtree
    double lmsubtree(mcinfo& mci); // Used to pool information together across nodes within a subtree
    double lmsubtreenode(mcinfo &mci); // Individual node in subtree -- applied to a node that is in a subtree but does not have the subtree info stored 

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
    
    // Was protected
    void local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir); //assumes brt.t is the root node
    void local_getsuff(diterator& diter, tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir); //assumes brt.t is the root node
    void local_mpigetsuff(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir); // birth version
    void local_mpigetsuff(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir); // death version
    void local_mpigetsuff_nodecases(tree::tree_p n, sinfo& sil, sinfo& sir, bool birthmove);

    
};

#endif