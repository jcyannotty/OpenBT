#include <iostream>

//header files from OpenBT 
#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"
#include "mxbrt.h"

//Include the Eigen header files
#include "Eigen/Dense"

//--------------------------------------------------
//a single iteration of the MCMC for brt model
void mxbrt::drawvec(rn& gen){
    //All the usual steps
    brt::drawvec(gen);

    // Update the in-sample predicted vector
    setf_mix();

    // Update the in-sample residual vector
    setr_mix();
}

//--------------------------------------------------
//slave controller for draw when using MPI
void mxbrt::drawvec_mpislave(rn& gen){
    //All the usual steps
    brt::drawvec_mpislave(gen);

    // Update the in-sample predicted vector
    setf_mix();

    // Update the in-sample residual vector
    setr_mix();
}

//--------------------------------------------------
//draw theta for a single bottom node for the brt model
vxd mxbrt::drawnodethetavec(sinfo& si, rn& gen){
    //initialize variables
    mxsinfo& mxsi=static_cast<mxsinfo&>(si);
    mxd I(k,k), Sig_inv(k,k), Sig(k,k), Ev(k,k), E(k,k), Sp(k,k);
    vxd muhat(k), evals(k), stdnorm(k), betavec(k);
    //double sig2 = (*ci.sigma)*(*ci.sigma); //error variance
    
    I = Eigen::MatrixXd::Identity(k,k); //Set identity matrix
    betavec = ci.beta0*Eigen::VectorXd::Ones(k); //Set the prior mean vector
    
    //Compute the covariance
    Sig_inv = mxsi.sumffw + (1.0/(ci.tau*ci.tau))*I; //Get inverse covariance matrix
    Sig = Sig_inv.llt().solve(I); //Invert Sig_inv with Cholesky Decomposition
    
    //Compute the mean vector
    muhat = Sig*(mxsi.sumfyw + (1.0/(ci.tau*ci.tau))*I*betavec); //Get posterior mean -- may be able to simplify this calculation (k*ci.beta0/(ci.tau*ci.tau)) 

    //Spectral Decomposition of Covaraince Matrix Sig -- maybe move to a helper function
    //--Get eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Sig); //Get eigenvectors and eigenvalues
    if(eigensolver.info() != Eigen::ComputationInfo::Success) abort(); //Checks if any errors occurred
    evals = eigensolver.eigenvalues(); //Get vector of eigenvalues
    Ev = eigensolver.eigenvectors(); //Get matrix of eigenvectors

    //--Get sqrt of eigen values and store into a matrix
    E = mxd::Identity(k,k)*evals; //Diagonal Matrix of eigen values
    E = E.array().sqrt(); //Diagonal Matrix of sqrt of eigen values

    //--Compute Spectral Decomposition
    Sp = Ev*E*Ev;

    //Generate MVN Random vector
    //--Get vector of standard normal normal rv's
    for(int i=0; i<k;i++){
        stdnorm(i) = gen.normal();
    }
    
    //--Generate the MVN vector
    return muhat + Sp*stdnorm;
}

//--------------------------------------------------
//lm: log of integrated likelihood, depends on prior and suff stats
double mxbrt::lm(sinfo& si){
    mxsinfo& mxsi=static_cast<mxsinfo&>(si);
    mxd Sig_inv(k,k), Sig(k,k), I(k,k);
    vxd beta(k); 
    double t2 = ci.tau*ci.tau;
    double suml; //sum of log determinent 
    double sumq; //sum of quadratic term

    I = mxd::Identity(k,k); //Set identity matrix
    beta = ci.beta0*vxd::Ones(k);

    //Get covariance matrix
    Sig_inv = mxsi.sumffw + I/t2; //get sig_inv matrix
    Sig = Sig_inv.llt().solve(I); //Get Sigma
    
    //Compute Log determinent
    mxd L(Sig.llt().matrixL()); //Cholesky decomp and store the L matrix
    suml = 2*(L.diagonal().array().log().sum()); //The log determinent is the same as 2*sum(log(Lii)) --- Lii = diag element of L

    //Now work on the exponential terms
    sumq = mxsi.sumyyw - (mxsi.sumfyw + beta/t2).transpose()*Sig*(mxsi.sumfyw + beta/t2);

    //print the mxinfo
    mxsi.print_mx();
    return 0.5*(suml - sumq);

    

    //cout << "msi.sumw=" << msi.sumw << " msi.sumwy=" << msi.sumwy << endl;
    //return -.5*log(k)+.5*msi.sumwy*msi.sumwy*t2/k;
}

/*
//Define the mxbrt class -- inherits from brt
class mxbrt : public brt{
    public:
        //classes: cinfo, tprior, and mcmcinfo. Only the cinfo is different from the brt class
        //cinfo = paramters for the end node model
        class cinfo{
            public:
                cinfo():beta0(1.0), tau(1.0), sigma(0) {} //beta0 = scalar in the prior mean vector, tau = prior stdev for tnode parameters, sigma = stdev of error 
                double beta0, tau;
                double* sigma; //use pointer since this will be changed as mcmc iterates
        };

        //constructors & destructors
        mxbrt():brt(){} 

        //methods
        //draw -- run a single iteration of the mcmc
        void draw(rn& gen){
            //Call the draw function from brt -- updates the tree and envokes function to draw new parameters for the mcmc iteration
            brt::draw(gen);

            //Update the in-sample predicted vector
            setf();

            //Update the in-sample residual vector
            setr();

        }

        //draw_mpislave

        //setci -- set the terminal node parameters (beta0, tau, and sigma in model mixing bart)
        void setci(double beta0, double tau, double* sigma){
            ci.beta0 = beta0;
            ci.tau = tau;
            ci.sigma = sigma;
        }

        //drawtheta -- this is an override that does not occur in mbrt. This is done because we need to return a vector rather than a scalar in drawnodetheta
        void drawtheta(sinfo& si, rn& gen){
            //Initialize a bottom node vector and a vector of type sinfo
            tree::npv bnv; 
            std::vector<sinfo*>& siv = newsinfovec(); //newsinfovec() returns new sinfo -- this initializes a new sinfo vector

            //Get all sufficient stats assigned to bnv..(?)
            allsuff(bnv, siv);
            
            //Check MPI
            #ifdef _OPENMPI
                mpi_resetrn(gen);
            #endif
            
            //Loop through bottom node vector and draw a parameter vector(!!)
            for(size_t i = 0; i<bnv.size(); i++){
                bnv[i]->setthetavec(drawnodethetavec(*(siv[i]),gen));
                delete siv[i]; 
            }

        }

        //drawnodetheta -- sample from the posterior of the terminal node parameters
        //virtual Eigen::VectorXd drawnodethetavec(sinfo& si, rn& gen){

        //}

        protected:
        //Model Information -- conditioning info = parameters and hyperparameters to condition on
        cinfo ci;

};

*/



