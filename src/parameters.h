// Creates a parameters class for calibration (and potentially other models/settings) where
// one needs to learn a parameter

#ifndef GUARD_parameters_h
#define GUARD_parameters_h

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rn.h"

#ifdef _OPENMPI
#   include <mpi.h>
#   include <crn.h>
#   define SIZE_UINT1 16  // sizeof(unsigned int)
#   define SIZE_UINT2 32  // sizeof(unsigned int)*2
#   define SIZE_UINT3 48  // sizeof(unsigned int)*3
#   define SIZE_UINT4 64  // sizeof(unsigned int)*4
#   define SIZE_UINT5 80  // sizeof(unsigned int)*5
#   define SIZE_UINT6 96  // sizeof(unsigned int)*6
#   define MPI_TAG_CP_PASSUNEW 100
#   define MPI_TAG_CP_ACCEPT 101
#   define MPI_TAG_CP_REJECT 102
#endif


class param{
    public:
        param():p(1),tc(1),rank(0),adaptcount(1),ucur(1,0),unew(1,0),acceptvec(1,0),rejectvec(1,0) {}
        param(size_t ip):p(ip),tc(1),rank(0),adaptcount(1),ucur(ip,0),unew(ip,0),acceptvec(ip,0),rejectvec(ip,0) {}
        ~param(){} // destructor

        // Objects
        size_t p; // number of parameters
        int tc;
        size_t rank;
        bool accept;
        size_t adaptcount;
        std::vector<double> ucur; // Current parameter vector u
        std::vector<double> unew; // New parameter vector u
        std::vector<size_t> acceptvec; // Accept counter
        std::vector<size_t> rejectvec; // Reject counter
        std::vector<std::string> propdistvec; // mh proposal distributions
        std::vector<double> propvec; // mh proposal vector (distance or sd depending on the distribution)
        std::vector<std::string> priordistvec; // Prior distributions
        std::vector<double> priorp1vec; // Prior parameter 1
        std::vector<double> priorp2vec; // Prior parameter 2
 
        // Methods
        void adapt();
        void drawnew(size_t ind,rn &gen); // Proposal for one parameter
        void drawnew(rn &gen); // Joint proposal
        void drawnew_mala(std::vector<double> grad,rn &gen); // proposal using mala-like idea
        void drawnew_mpi(size_t ind,rn &gen); // Proposal for one parameter mpi version
        void drawnew_mpi(rn &gen); // Joint proposal mpi version
        //void drawnew_mala_mpi(std::vector<double> grad,rn &gen); // proposal using mala-like idea
        double lm(double wssr_cur, double wssr_new);
        double logprp_mala(std::vector<double> u1,std::vector<double> u2, std::vector<double> grad); // log proposal probability for mala move
        void mhstep(double cursumwr2, double newsumwr2, rn &gen);
        void setmpirank(int rank) {this->rank = rank;}  //only needed for MPI
        void setpriors(std::vector<std::string> prdist,std::vector<double> prp1, std::vector<double> prp2);
        void setproposals(std::vector<std::string> propdist, std::vector<double> prop);
        void settc(int tc) {this->tc = tc;}  // this is numslaves for MPI, or numthreads for OPEN_MP
        void setucur(std::vector<double> uc){this->ucur = uc;this->unew = uc;}
        void updatex(std::vector<double> &x, std::vector<size_t> ucols, size_t pxu, size_t n);

   
};


#endif