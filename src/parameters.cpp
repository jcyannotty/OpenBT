#include <iostream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rn.h"
#include "crn.h"
#include "parameters.h"

using std::cout;
using std::endl;

//----------------------------------------------
// Adapt step for proposals
void param::adapt(){
    double accrate;
    double gamma1, gamma2;
    
    if(propdistvec[0] == "mala"){
        //if(rank==0) cout << "adapt mala" << endl;
        for(size_t j=0;j<p;j++) {
            if(rank==0) std::cout << "Acceptance rate=" << ((double)acceptvec[j])/((double)(acceptvec[j]+rejectvec[j]));
            accrate=((double)acceptvec[j])/((double)(acceptvec[j]+rejectvec[j]));
            gamma1 = 1/(pow(adaptcount,0.5));
            gamma2 = 2.5*gamma1;
            cout << "----gamma2 = " << gamma2 << "----";
            //if(rank==0) std::cout << " (propwidth =  " << propvec[j] << ")";
            propvec[j] = sqrt(exp(2*log(propvec[j]) + gamma2*(accrate-0.50)));
            if(rank==0) std::cout << " (adapted propwidth to " << propvec[j] << ")" << std::endl;
        }
    }else{
        for(size_t j=0;j<p;j++) {
            if(rank==0) std::cout << "Acceptance rate=" << ((double)acceptvec[j])/((double)(acceptvec[j]+rejectvec[j]));
            accrate=((double)acceptvec[j])/((double)(acceptvec[j]+rejectvec[j]));
            if(accrate>0.29 || accrate<0.19) propvec[j]*=accrate/0.24;
            if(rank==0) std::cout << " (adapted propwidth to " << propvec[j] << ")" << std::endl;
        }
    }
    
    adaptcount+=1;
    std::fill(acceptvec.begin(),acceptvec.end(),0);
    std::fill(rejectvec.begin(),rejectvec.end(),0);
}


//----------------------------------------------
// Draw new parameter -- marginal update
void param::drawnew(size_t ind, rn &gen){
    double u0;
    u0 = 0;
    cout << "here by mistake" << endl;
}

//----------------------------------------------
// Draw new parameter -- joint update
void param::drawnew(rn &gen){
    unew.clear();
    unew.resize(p);
    for(size_t i=0;i<p;i++){
        if(propdistvec[i] == "uniform"){
            unew[i] = ucur[i] + (gen.uniform() - 0.5)*propvec[i];
            //std::cout << "unew[i] = " << unew[i] << std::endl; 
        }else if(propdistvec[i] == "normal"){
            unew[i] = ucur[i] + gen.normal()*propvec[i];
        }
    }
    // Pass the unew vector (as an array) if using mpi  
#ifdef _OPENMPI
    if(rank==0){ //should always be true when using mpi
        char buffer[SIZE_UINT3*p];
        int position=0;
        MPI_Request *request=new MPI_Request[tc];
        const int tag=MPI_TAG_CP_PASSUNEW;
        double unarray[p];

        // Cast vector to array
        copy(unew.begin(),unew.end(),unarray);

        // Pack and send info to the slaves
        MPI_Pack(&unarray,p,MPI_DOUBLE,buffer,SIZE_UINT3*p,&position,MPI_COMM_WORLD);
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(buffer,SIZE_UINT3*p,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);

        delete[] request;
    }
#endif
}    


//----------------------------------------------
// drawnew_mpi for individual parameter updates
void param::drawnew_mpi(size_t ind, rn &gen){
    cout << "Not yet implemented..." << endl;
}

//----------------------------------------------
// drawnew_mpi
void param::drawnew_mpi(rn &gen){
#ifdef _OPENMPI
    int buffer_size = SIZE_UINT3*p;
    char buffer[buffer_size];
    int position=0;
    MPI_Status status;
 
    // MPI receive the proposed u vector.
    MPI_Recv(buffer,buffer_size,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
    double unarray[p];
    MPI_Unpack(buffer,buffer_size,&position,&unarray,p,MPI_DOUBLE,MPI_COMM_WORLD);
 
    unew.clear();
    for(size_t j=0;j<p;j++){unew.push_back(unarray[j]);}
#endif    
}


//----------------------------------------------
// Draw new parameter -- joint update using MALA-like method
void param::drawnew_mala(std::vector<double> grad, rn &gen){
    unew.clear();
    unew.resize(p);
    for(size_t i=0;i<p;i++){
        // Get proposed u value
        unew[i] = ucur[i] + propvec[i]*grad[i] + 2*propvec[i]*gen.normal();         
    }
    // Pass the unew vector (as an array) if using mpi  
#ifdef _OPENMPI
    if(rank==0){ //should always be true when using mpi
        char buffer[SIZE_UINT3*p];
        int position=0;
        MPI_Request *request=new MPI_Request[tc];
        const int tag=MPI_TAG_CP_PASSUNEW;
        double unarray[p];

        // Cast vector to array
        copy(unew.begin(),unew.end(),unarray);

        // Pack and send info to the slaves
        MPI_Pack(&unarray,p,MPI_DOUBLE,buffer,SIZE_UINT3*p,&position,MPI_COMM_WORLD);
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(buffer,SIZE_UINT3*p,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);

        delete[] request;
    }
#endif
}    
 

//----------------------------------------------
// lm for calibration
double param::lm(double cursumwr2, double newsumwr2){
    double lprior = 0.0, ltemp = 0.0;
    double lalpha;

    // Get the log likelihood ratio
    lalpha = 0.5*(cursumwr2 - newsumwr2);

    // Get the logprior ratio
    for(size_t i=0;i<p;i++){
        if(priordistvec[i]=="uniform"){
            lprior += 0.0;
            if(unew[i]<priorp1vec[i] || unew[i]>priorp2vec[i]){
                lprior=-std::numeric_limits<double>::infinity();
            }
        }else if(priordistvec[i]=="normal"){
            lprior += 0.5*(ucur[i]-priorp1vec[i])*(ucur[i]-priorp1vec[i])/(priorp2vec[i]*priorp2vec[i]);
            lprior += -0.5*(unew[i]-priorp1vec[i])*(unew[i]-priorp1vec[i])/(priorp2vec[i]*priorp2vec[i]);
        }
    }
    // Get the output lm
    return lprior + lalpha;
}

//----------------------------------------------
// log proposal probability for mala move: u1 --> u2
// u1 -- conditional on u1
// u2 -- the move i.e. u1 --> u2
double param::logprp_mala(std::vector<double> u1,std::vector<double> u2, std::vector<double> grad){
    // Compute the normal kernel (removing 1/2 here because 1/2 is taken out in mhstep as of now)
    double psum = 0;
    double ptemp = 0;
    for(size_t i=0;i<p;i++){
        ptemp = u2[i]-u1[i] - propvec[i]*grad[i];
        psum += psum + ptemp*ptemp;
    }
    return psum;
}


//----------------------------------------------
// MH step
void param::mhstep(double csumwr2, double nsumwr2, rn &gen){
#ifdef _OPENMPI
    if(rank>0){
        // Sum the suff stats then get accept/reject status
        MPI_Status status;
        MPI_Reduce(&csumwr2,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&nsumwr2,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Recv(NULL,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        // Carry out the accept/reject step
        if(status.MPI_TAG==MPI_TAG_CP_ACCEPT) {
            for(size_t j=0;j<p;j++) acceptvec[j]++;
            ucur = unew;
            accept=true;
        }else{
            for(size_t j=0;j<p;j++) rejectvec[j]++;
            accept=false;
        }
    }           

    if(rank==0){
        // Reduce the suff stats
        MPI_Request *request = new MPI_Request[tc];
        MPI_Reduce(MPI_IN_PLACE,&csumwr2,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE,&nsumwr2,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

        // Do mhstep
        double lmout = lm(csumwr2,nsumwr2);
        double alpha = gen.uniform();
        if(log(alpha)<lmout){
            // accept
            ucur = unew;
            accept = true;
            for(size_t j=0;j<p;j++) acceptvec[j]++;
            const int tag=MPI_TAG_CP_ACCEPT;
            for(size_t k=1; k<=(size_t)tc; k++){
                MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
            }
        }else{ 
            // reject
            const int tag=MPI_TAG_CP_REJECT;
            accept = false;
            for(size_t k=1; k<=(size_t)tc; k++) {
                MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
            }
            for(size_t j=0;j<p;j++) rejectvec[j]++;
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;
    }
#else
    // Do mhstep
    double lmout = lm(csumwr2,nsumwr2);
    double alpha=gen.uniform();
    if(log(alpha)<lmout){
        // accept
        ucur = unew;
        accept=true;
        for(size_t j=0;j<p;j++) acceptvec[j]++;
    }else{ 
        // reject
        accept=false;
        for(size_t j=0;j<p;j++) rejectvec[j]++;
    }
#endif
}

//----------------------------------------------
// Set priors
void param::setpriors(std::vector<std::string> prdist, std::vector<double> prp1, std::vector<double> prp2){
    for(size_t i=0;i<prdist.size();i++){
        priordistvec.push_back(prdist[i]);
        priorp1vec.push_back(prp1[i]);
        priorp2vec.push_back(prp2[i]);
    }
}

//----------------------------------------------
// Set proposals
void param::setproposals(std::vector<std::string> propdist, std::vector<double> prop){
    for(size_t i=0;i<propdist.size();i++){
        propdistvec.push_back(propdist[i]);
        propvec.push_back(prop[i]);
    }
}

//----------------------------------------------
// Update x vector with unew, assumes the ucolumns are after all the regular x columns
// ucols = the index of each u parameter (0,1,2,...,p), given p+1 total parameters
void param::updatex(std::vector<double> &x, std::vector<size_t> ucols, size_t pxu, size_t n){
    size_t uidx = pxu - ucols.size();
    size_t uc;
    //cout << "pxu = " << pxu << endl;
    for(size_t i=0;i<n;i++){
        for(size_t j=0;j<ucols.size();j++){
            uc = ucols[j];
            //cout << "xold = " << x[i*pxu + uc] << endl;
            x[i*pxu + uc] = unew[j];
            //cout << "xnew = " << x[i*pxu + uc] << endl;
        }
    }
}