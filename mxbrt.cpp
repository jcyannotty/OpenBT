#include <iostream>

//header files from OpenBT 
#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"

#include "mbrt.h"

//Include the Eigen header files
#include "Eigen/Dense"

//Define the mxinfo class -- inherits from sinfo
class mxinfo : public sinfo{
    public:
        //Constructors:
        mxinfo():sinfo(),k(1),sumfft(Eigen::MatrixXd::Zero(k,k)), sumfr(Eigen::VectorXd::Zero(k)) {} //Initialize mxinfo with default settings
        mxinfo(sinfo& is, int k0, Eigen::MatrixXd sff, Eigen::VectorXd sfr):sinfo(is), k(k0), sumfft(sff), sumfr(sfr) {} //Construct mxinfo instance with values -- need to use references
        mxinfo(const mxinfo& is):sinfo(is),k(is.k),sumfft(is.sumfft),sumfr(is.sumfr){} //Input object of mxinfro (is) and instantiate new mxinfo 

        virtual ~mxinfo() {} //free memory -- need to figure out how this works -- can't use the first constructor in int main 

        //Initialize sufficient stats
        Eigen::MatrixXd sumfft; //Computes F^t*F by summing over fi*fi^t for the observations in each tnode (fi = vector of dimension K)
        Eigen::VectorXd sumfr; //Computes F^t*R by summing over fi*ri for observations in each tnode (fi = vector and ri = scalar) 
        int k; //number of models, so equal to number of columns in sumfft. This is needed in order to initialize Zero matrix/vector in eigen

        //Define Operators -- override from sinfo class
        //Compound addition operator - used when adding sufficient statistics
        virtual sinfo& operator+=(const sinfo& rhs){
            sinfo::operator+=(rhs); 
            const mxinfo& mrhs = static_cast<const mxinfo&>(rhs); //Cast rhs as an mxinfo instance.  
            sumfft += mrhs.sumfft;
            sumfr += mrhs.sumfr;
            return *this; //returning *this should indicate that we return updated sumfft and sumfr while also using a pointer
        }

        //Compound assignment operator for sufficient statistics
        virtual sinfo& operator=(const sinfo& rhs){
            if(&rhs != this){
                sinfo::operator=(rhs); //--Figure out what this line does
                const mxinfo& mrhs=static_cast<const mxinfo&>(rhs);
                this->sumfft = mrhs.sumfft; 
                this->sumfr = mrhs.sumfr;
                this->k = mrhs.k; //May not need this assignment
                return *this; 
            }
            return *this; //returning *this should indicate that we return updated sumfft and sumfr while also using a pointer
        }

        //Addition operator -- defined in terms of the compund operator above. Use for addition across two instances of mxinfo
        const mxinfo operator+(const mxinfo& other) const{
            mxinfo result = *this;
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
            /*
            for(size_t i = 0; i<bnv.size(); i++){
                bnv[i]->setthetavec(drawnodethetavec(*(siv[i]),gen));
                delete siv[i]; 
            }
            */

        }

        //drawnodetheta -- sample from the posterior of the terminal node parameters
        //virtual Eigen::VectorXd drawnodethetavec(sinfo& si, rn& gen){

        //}

        protected:
        //Model Information -- conditioning info = parameters and hyperparameters to condition on
        cinfo ci;

};


class test{
    public:
        test(): tt(0.0){}
        double tt;
};

int main(){
    //Initialize matrix and vector
    Eigen::MatrixXd F;
    Eigen::VectorXd v;
    int k = 4;

    F = Eigen::MatrixXd::Random(k,k);
    v = Eigen::VectorXd::Random(k);

    //Work with sinfo object
    sinfo s;
    std::cout << s.n << std::endl; //Prints out 0 as expected

    //Try to create mxinfo objects
    mxinfo mx1(s, k, F, v); //Constructor 2
    mxinfo mx2(mx1); //Constructor 3

    //See what they print
    mx1.print_mx();
    mx2.print_mx();

    //Work with operators
    std::cout << "Compound Addition Operator" << std::endl;
    mx1 += mx2;
    mx1.print_mx();

    std::cout << "Addition Operator" << std::endl;    
    //Try the other addition operator  
    mxinfo mx3 = mx1.operator+(mx2); //this can be used to create a new instance of mxinfo 
    mx3.print_mx();

    //Make a tree -- see if this throws an error
    double tv = 0.1;
    tree t(tv);

    test temp;
    std::cout << temp.tt << std::endl;

    return 0;

}



