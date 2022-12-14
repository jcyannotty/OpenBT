// Calibration helper functions
#ifndef GUARD_cal
#define GUARD_cal

#include <chrono>
#include <iostream>
#include <string>
#include <ctime>
#include <sstream>

#include <fstream>
#include <vector>
#include <limits>

#include "Eigen/Dense"
#include <Eigen/StdVector>

#include "crn.h"
#include "brt.h"

#ifdef _OPENMPI
    #include <mpi.h>
    #define MPI_TAG_CAL_ACCEPT 9001
    #define MPI_TAG_CAL_REJECT 9002
#endif

// Update x with a new set of calibration parameters
void updatex(std::vector<double> &x, std::vector<size_t> ucols,std::vector<double> unew, size_t p, size_t n){
    size_t uidx = p - ucols.size();
    for(size_t i=0;i<n;i++){
        for(size_t j=0;j<ucols.size();j++){
            x[i*uidx+j] = unew[j];
        }
    }
}


// Get new uvec
std::vector<double> getnewu(std::vector<double> uc, std::vector<double> propwidth, rn& gen){
    std::vector<double> un(uc.size());
    for(size_t i=0;i<uc.size();i++){
        un[i] = uc[i] + propwidth[i]*(gen.uniform()-0.5);
    }
    return un;
}


// Log metropolis hastings step
void logmhstep(double newsumwr2,double oldsumwr2,std::vector<double> &unew,std::vector<double> &uold, 
                std::vector<size_t> acceptu,std::vector<size_t> rejectu,
                rn& gen, int mpirank, int tc){
    double lalpha;
    size_t pu = uold.size();
    lalpha = 0.5*(oldsumwr2 - newsumwr2);

#ifdef _OPENMPI
    if(mpirank>0){
        MPI_Status status;
        if(status.MPI_TAG==MPI_TAG_CAL_ACCEPT) {
            uold = unew;
            for(size_t j=0;j<pu;j++) acceptu[j]++;
        }else{
            for(size_t j=0;j<pu;j++) rejectu[j]++;
        }
    }
    
    if(mpirank==0) {
        MPI_Request *request = new MPI_Request[tc];          
        double alpha=gen.uniform();
        if(log(alpha)<lalpha){ //accept
            uold = unew;
            for(size_t j=0;j<pu;j++) acceptu[j]++;
            const int tag=MPI_TAG_CAL_ACCEPT;
            for(size_t k=1; k<(size_t)tc; k++) {
                MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
            }
        }else{ //reject
            const int tag=MPI_TAG_CAL_REJECT;
            for(size_t k=1; k<(size_t)tc; k++) {
                MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
            }
            for(size_t j=0;j<pu;j++) rejectu[j]++;
        }
        MPI_Waitall(tc-1,request,MPI_STATUSES_IGNORE);
        delete[] request;
    }
#endif
}

//--------------------------------------------------
//local_getsumr2 -- performs the calculation for getting the resid sum of squares
void local_getsumwr2(double &sumwr2, std::vector<double> &y, std::vector<double> &yhat, double* sig, size_t n){
    double resid;
    //Get sum of residuals squared
    for(size_t i = 0; i<n;i++){
        resid = (y[i]-yhat[i])/sig[i];
        sumwr2 = sumwr2 + resid*resid;
        //std::cout << resid[i] << " --- " << resid[i]*resid[i] << std::endl;
    }
} 

//--------------------------------------------------
//local_mpi_reduce_getsumr2 -- the MPI communication part of local_mpi_getsrumwr2.
void local_mpi_reduce_getsumwr2(double &sumwr2, int mpirank, int tc)
{
    #ifdef _OPENMPI
        double sr2 = sumwr2; //this should be zero on root 
        if(mpirank==0) {
            MPI_Status status;
            double tempsr2;
            
            //Receive, Send, and update sr2
            for(size_t i=1; i<=(size_t)tc; i++) {
                MPI_Recv(&tempsr2,1,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
                sr2 += tempsr2;
            }

            MPI_Request *request=new MPI_Request[tc];
            for(size_t i=1; i<=(size_t)tc; i++) {
                MPI_Isend(&sr2,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
            }

            //set sumr2 to the value
            sumwr2 = sr2;

            MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
            delete[] request;
        } else {
            //Send and receive sumr2
            MPI_Request *request=new MPI_Request;
            MPI_Status status;

            MPI_Isend(&sr2,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
            MPI_Wait(request,MPI_STATUSES_IGNORE);
            delete request;

            MPI_Recv(&sr2,1,MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            
            //update sumr2 to the value
            sumwr2 = sr2;
        }
    #endif
}

//--------------------------------------------------
//local_mpi_getsumwr2
void local_mpi_getsumwr2(double &sumwr2, std::vector<double> &y, std::vector<double> &yhat, double* sig, size_t n, int mpirank, int tc){
    #ifdef _OPENMPI
        local_mpi_reduce_getsumwr2(sumwr2,mpirank,tc);
    #else
        local_getsumwr2(sumwr2,y,yhat,sig,n);
        local_mpi_reduce_getsumwr2(sumwr2,mpirank,tc);
    #endif
}

//--------------------------------------------------
//getsumwr2 -- calls local functions similar to allsuff
void getsumwr2(double &sumwr2, std::vector<double> &y, std::vector<double> &yhat, double* sig, size_t n, int mpirank, int tc){
    #ifdef _OPENMPI
        local_mpi_getsumwr2(sumwr2,y,yhat,sig,n,mpirank,tc);
    #else
        local_getsumwr2(sumwr2,y,yhat,sig,n);
    #endif
}



/*
if((i+1)%adaptevery==0) {
        double accrate;
        for(size_t j=0;j<pthetas;j++) {
        if(mpirank==0) cout << "Acceptance rate=" << ((double)accept_thetas[j])/((double)(accept_thetas[j]+reject_thetas[j]));
        accrate=((double)accept_thetas[j])/((double)(accept_thetas[j]+reject_thetas[j]));
        if(accrate>0.29 || accrate<0.19) propwidth[j]*=accrate/0.24;
        if(mpirank==0) cout << " (adapted propwidth to " << propwidth[j] << ")" << endl;
        }
        std::fill(accept_thetas.begin(),accept_thetas.end(),0);
        std::fill(reject_thetas.begin(),reject_thetas.end(),0);
    }
*/
#endif