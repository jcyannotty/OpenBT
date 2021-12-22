#include <iostream>

//header files from OpenBT 
#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"
#include "mxbrt.h"

//Include the Eigen header files
#include "Eigen/Dense"



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



