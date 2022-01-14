#include <iostream>

//header files from OpenBT 
#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"
#include "mxbrt.h"

//Include the Eigen header files
#include "Eigen/Dense"
#include <Eigen/StdVector>

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
    E = mxd::Zero(k,k); //Set as matrix of 0's
    E.diagonal() = evals.array().sqrt(); //Diagonal Matrix of sqrt of eigen values
    
    //--Compute Spectral Decomposition
    Sp = Ev*E*Ev.transpose();

    //Generate MVN Random vector
    //--Get vector of standard normal normal rv's
    for(size_t i=0; i<k;i++){
        stdnorm(i) = gen.normal(); 
    }
    //Print out matrix algebra step-by-step
    /*
    std::cout << "\nAll matrix Calculations:" << std::endl;
    std::cout << "Sig_inv = \n" << Sig_inv << std::endl;
    std::cout << "\n Sig = \n" << Sig << std::endl;
    std::cout << "\n muhat = \n" << muhat << std::endl;
    std::cout << "\n Ev = \n" << Ev << std::endl;
    std::cout << "\n evals = \n" << evals << std::endl;
    std::cout << "\n E = \n" << E << std::endl;
    std::cout << "\n Sp = \n" << Sp << std::endl;
    std::cout << "\n thetavec = " << std::endl;
    */

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
    Sig = mxsi.sumffw + I/t2; //get Sig matrix
    Sig_inv = Sig.llt().solve(I); //Get Sigma_inv
    
    //Compute Log determinent
    mxd L(Sig.llt().matrixL()); //Cholesky decomp and store the L matrix
    suml = 2*(L.diagonal().array().log().sum()); //The log determinent is the same as 2*sum(log(Lii)) --- Lii = diag element of L

    //Now work on the exponential terms
    sumq = mxsi.sumyyw - (mxsi.sumfyw + beta/t2).transpose()*Sig_inv*(mxsi.sumfyw + beta/t2) + k*ci.beta0*ci.beta0/t2;
  
    //print the mxinfo
    //mxsi.print_mx();
    return -0.5*(suml + sumq + k*log(t2));

}

//--------------------------------------------------
//Add in an observation, this has to be changed for every model.
//Note that this may well depend on information in brt with our leading example
//being double *sigma in cinfo for the case of e~N(0,sigma_i^2).
// Note that we are using the training data and the brt object knows the training data
//     so all we need to specify is the row of the data (argument size_t i).
void mxbrt::add_observation_to_suff(diterator& diter, sinfo& si){
    //Declare variables 
    mxsinfo& mxsi=static_cast<mxsinfo&>(si);
    double w, yy;
    mxd ff(k,k);
    vxd fy(k);
    
    //Assign values
    w=1.0/(ci.sigma[*diter]*ci.sigma[*diter]);
    ff = (*fi).row(*diter).transpose()*(*fi).row(*diter);
    fy = (*fi).row(*diter).transpose()*diter.gety();
    yy = diter.gety()*diter.gety();

    //std::cout << "---------------------" << std::endl;
    //std::cout << "Obs i = " << *diter << std::endl;
    //std::cout << "y = " << diter.gety() << std::endl;
    //std::cout << "fi = " << (*fi).row(*diter) << std::endl;

    //Update sufficient stats for nodes
    mxsi.n+=1;
    mxsi.sumffw+=w*ff;
    mxsi.sumfyw+=w*fy;
    mxsi.sumyyw+=w*yy;

    //Print to check
    //mxsi.print_mx();
}

//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
void mxbrt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir){
#ifdef _OPENMPI
   mxsinfo& msil=static_cast<mxsinfo&>(sil);
   mxsinfo& msir=static_cast<mxsinfo&>(sir);
   if(rank==0) { // MPI receive all the answers from the slaves
      MPI_Status status;
      mxsinfo& tsil = (mxsinfo&) *newsinfo();
      mxsinfo& tsir = (mxsinfo&) *newsinfo();
      char buffer[SIZE_UINT6];
      int position=0;
      unsigned int ln,rn;
      int mat_sz = tsil.sumffw.size(); //matrix size (number of elements) 
      int vec_sz = tsil.sumfyw.size(); //vector dimension (number of elements)
      
      //Cast the matrices and vectors to arrays
      double sumffw_rarray[k*k], sumffw_larray[k*k]; //arrays for right and left matrix suff stats
      double sumfyw_rarray[k], sumfyw_larray[k]; //arrays for right and left vector suff stats
      matrix_to_array(tsil.sumffw, &sumffw_larray[0]); //function defined in brtfuns
      matrix_to_array(tsir.sumffw, &sumffw_rarray[0]); //function defined in brtfuns
      std::copy(std::begin(tsil.sumfyw.array()), std::end(tsil.sumfyw.array()), std::begin(sumfyw_larray)); //use std functions for vectors
      std::copy(std::begin(tsir.sumfyw.array()), std::end(tsir.sumfyw.array()), std::begin(sumfyw_rarray));

      for(size_t i=1; i<=(size_t)tc; i++) {
         position=0;
         MPI_Recv(buffer,SIZE_UINT6,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&sumffw_larray,mat_sz,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&sumffw_rarray,mat_sz,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&sumfyw_larray,vec_sz,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&sumfyw_rarray,vec_sz,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&tsil.sumyyw,1,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&tsir.sumyyw,1,MPI_DOUBLE,MPI_COMM_WORLD);

         tsil.n=(size_t)ln;
         tsir.n=(size_t)rn;
         msil+=tsil;
         msir+=tsir;
      }
      delete &tsil;
      delete &tsir;
   }
   else // MPI send all the answers to root
   {
      char buffer[SIZE_UINT6];
      int position=0;  
      unsigned int ln,rn;

      int mat_sz = tsil.sumffw.size(); //matrix size (number of elements) 
      int vec_sz = tsil.sumfyw.size(); //vector dimension (number of elements)
      
      //Cast the matrices and vectors to arrays
      double sumffw_rarray[k*k], sumffw_larray[k*k]; //arrays for right and left matrix suff stats
      double sumfyw_rarray[k], sumfyw_larray[k]; //arrays for right and left vector suff stats
      matrix_to_array(tsil.sumffw, &sumffw_larray[0]); //function defined in brtfuns
      matrix_to_array(tsir.sumffw, &sumffw_rarray[0]); //function defined in brtfuns
      std::copy(std::begin(tsil.sumfyw.array()), std::end(tsil.sumfyw.array()), std::begin(sumfyw_larray)); //use std functions for vectors
      std::copy(std::begin(tsir.sumfyw.array()), std::end(tsir.sumfyw.array()), std::begin(sumfyw_rarray));

      ln=(unsigned int)msil.n;
      rn=(unsigned int)msir.n;
        /*
        std::cout << "Address: " << &msil.sumffw << std::endl;
        std::cout << "Address: " << msil.sumffw.data() << std::endl;
        std::cout << "Address: " << &msil.sumfyw << std::endl;
        std::cout << "Address: " << msil.sumfyw.data() << std::endl;
        std::cout << "Address: " << &msil.sumyyw << std::endl;
        std::cout << "Address: " << &msir.sumffw << std::endl;
        std::cout << "Address: " << msir.sumffw.data() << std::endl;
        std::cout << "Address: " << &msir.sumfyw << std::endl;
        std::cout << "Address: " << msir.sumfyw.data() << std::endl;
        std::cout << "Address: " << &msir.sumyyw << std::endl;
        */
      MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&sumffw_larray,mat_sz,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&sumffw_rarray,mat_sz,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&sumfyw_larray,vec_sz,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&sumfyw_rarray,vec_sz,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&msil.sumyyw,1,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&msir.sumyyw,1,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);

      MPI_Send(buffer,SIZE_UINT6,MPI_PACKED,0,0,MPI_COMM_WORLD);
   }
#endif   
}

//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void mxbrt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv){
#ifdef _OPENMPI
    //Construct containers for suff stats -- each ith element in the container is a suff stat for the ith node
    unsigned int nvec[siv.size()];
    double sumyywvec[siv.size()];
    double sumffwvec[k*k*siv.size()]; //An array that contains siv.size() matricies of kXk dimension that are flattened (by row).
    double sumfywvec[k*siv.size()]; //An array that contains siv.size() vectors of k dimension.
    
    std::vector<mxd, Eigen::aligned_allocator<mxd>> sumffwvec_em(siv.size()); //Vector of Eigen Matrixes kXk dim
    std::vector<vxd, Eigen::aligned_allocator<vxd>> sumfywvec_ev(siv.size()); //Vector of Eigen Vectors k dim
    
    for(size_t i=0;i<siv.size();i++) { // on root node, these should be 0 because of newsinfo().
        mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
        nvec[i]=(unsigned int)mxsi->n;    // cast to int
        sumyywvec[i]=mxsi->sumyyw;
        sumfywvec_ev[i]=mxsi->sumfyw;
        sumffwvec_em[i]=mxsi->sumffw;

        //Cast the current matrix to an array -- keep in mind to location given the stat for each node is stacked in the same array
        matrix_to_array(sumffwvec_em[i], &sumffwvec[(i*k*k]);

        //cast the current vecotor to an array
        std::copy(std::begin(sumfywvec_ev[i].array()), std::end(sumfywvec_ev[i].array()), &sumfywvec[i*k]);
    }

    // MPI sum
    // MPI_Allreduce(MPI_IN_PLACE,&nvec,siv.size(),MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE,&sumwvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE,&sumwyvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    if(rank==0) {
        //std::cout << "Enter with rank 0 ---- " << std::endl;
        MPI_Status status;
        unsigned int tempnvec[siv.size()];
        double tempsumyywvec[siv.size()];
        double tempsumffwvec[k*k*siv.size()]; //An array that contains siv.size() matricies of kXk dimension that are flattened (by row).
        double tempsumfywvec[k*siv.size()]; //An array that contains siv.size() vectors of k dimension.
        
        // receive nvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempnvec,siv.size(),MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++){
                nvec[j]+=tempnvec[j];
            }
        }
        MPI_Request *request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to mxsi
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            mxsi->n=(size_t)nvec[i];    // cast back to size_t
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        // receive sumyywvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumyywvec,siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<siv.size();j++)
                sumyywvec[j]+=tempsumyywvec[j];
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumyywvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to mxsi
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            mxsi->sumyyw=sumyywvec[i];
            //std::cout << "Sumyyw[" << i << "] = " << sumyywvec[i] << std::endl;
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        // receive sumfywvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumfywvec,k*siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<k*siv.size();j++){
                sumfywvec[j]+=tempsumfywvec[j]; //add temp vector to the sufficient stat
            }
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumfywvec,k*siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        
        // cast back to mxsi
        Eigen::VectorXd tempsumfyw(k); //temp vector
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            tempsumfyw = Eigen::VectorXd::Zero(k); //initialize as zero vector before populating each node's suff stat 
            
            //Turn the ith section of k elements in the array into an Eigen VextorXd
            for(size_t l=0; l<k; l++){
                tempsumfyw(l) = sumfywvec[i*siv.size()+l] 
            }
            mxsi->sumfyw=tempsumfyw;
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;

        // receive sumffwvec, update and send back.
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Recv(&tempsumffwvec,k*k*siv.size(),MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            for(size_t j=0;j<k*k*siv.size();j++)
                sumffwvec[j]+=tempsumffwvec[j];
        }
        request=new MPI_Request[tc];
        for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(&sumffwvec,k*k*siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
        }
        // cast back to mxsi
        Eigen::MatrixXd tempsumffw(k,k); //temp matrix
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            tempsumffw = Eigen::MatrixXd::Zero(k,k); //initialize as zero matrix before populating each node's suff stat
            
            //cast the specific k*k elements from the array into the suff stat matrix
            array_to_matrix(tempsumffw, &summffwvec[i*k*k]) 
            mxsi->sumffw=tempsumffw;
        }
        MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
        delete[] request;
    }   
    else {
        //std::cout << "Enter with rank " << rank << " ---- " << std::endl;
        MPI_Request *request=new MPI_Request;
        MPI_Status status;

        // send/recv nvec      
        MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&nvec,siv.size(),MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumyywvec, update nvec, receive sumyywvec
        request=new MPI_Request;
        MPI_Isend(&sumyywvec,siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to mxsi
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            mxsi->n=(size_t)nvec[i];    // cast back to size_t
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumyywvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumfywvec, update sumyywvec, receive sumfywvec
        request=new MPI_Request;
        MPI_Isend(&sumfywvec,k*siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to mxsi
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            mxsi->sumyyw=sumyywvec[i];
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumfywvec,k*siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // send sumffwvec, update sumfywvec, receive sumffwvec
        request=new MPI_Request;
        MPI_Isend(&sumffwvec,k*k*siv.size(),MPI_DOUBLE,0,0,MPI_COMM_WORLD,request);
        // cast back to mxsi
        Eigen::VectorXd tempsumfyw(k); //temp vector
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            tempsumfyw = Eigen::VectorXd::Zero(k); //initialize as zero vector before populating each node's suff stat 
            
            //Turn the ith section of k elements in the array into an Eigen VextorXd
            for(size_t l=0; l<k; l++){
                tempsumfyw(l) = sumfywvec[i*siv.size()+l] 
            }
            mxsi->sumfyw=tempsumfyw;
        }
        MPI_Wait(request,MPI_STATUSES_IGNORE);
        delete request;
        MPI_Recv(&sumffwvec,k*k*siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        // update sumffwvec
        // cast back to mxsi
        Eigen::MatrixXd tempsumffw(k,k); //temp matrix
        for(size_t i=0;i<siv.size();i++) {
            mxsinfo* mxsi=static_cast<mxsinfo*>(siv[i]);
            tempsumffw = Eigen::MatrixXd::Zero(k,k); //initialize as zero matrix before populating each node's suff stat
            
            //cast the specific k*k elements from the array into the suff stat matrix
            array_to_matrix(tempsumffw, &summffwvec[i*k*k]) 
            mxsi->sumffw=tempsumffw;
        }
    }

    // cout << "reduced:" << siv[0]->n << " " << siv[1]->n << endl;
#endif
}

//--------------------------------------------------
//Print mxbrt object
void mxbrt::pr_vec()
{
   std::cout << "***** mxbrt object:\n";
   std::cout << "Conditioning info:" << std::endl;
   std::cout << "   mean:   tau =" << ci.tau << std::endl;
   std::cout << "   mean:   beta0 =" << ci.beta0 << std::endl;
   if(!ci.sigma)
     std::cout << "         sigma=[]" << std::endl;
   else
     std::cout << "         sigma=[" << ci.sigma[0] << ",...," << ci.sigma[di->n-1] << "]" << std::endl;
   brt::pr_vec();
}

//--------------------------------------------------
//Sample from the variance posterior -- using Gibbs sampler under a homscedastic variance assumption
void mxbrt::drawsigma(rn& gen){
    double sumr2 = 0;
    int n;
    n = resid.size();

    //Get sum of residuals squared
    for(int i = 0; i<n;i++){
        sumr2 = sumr2 + resid[i]*resid[i];
        //std::cout << resid[i] << " --- " << resid[i]*resid[i] << std::endl;
    }
    //Get nu*lambda and nu
    double nulampost=ci.nu*ci.lambda+sumr2;
    double nupost = n + (int)ci.nu;

    //Generate new inverse scaled chi2 rv
    gen.set_df(nupost); //set df's
    double newsig = sqrt((nulampost)/gen.chi_square());
    for(int i = 0;i<n;i++){
        ci.sigma[i] = newsig; 
    }

    /*
    std::cout << "n=" << n << " -- sumr2=" << sumr2 << std::endl;
    std::cout << "Before --- *ci.sigma = " <<  *(ci.sigma) << std::endl;
    std::cout << "Chi2 = " <<  gen.chi_square() << std::endl;
    std::cout << "nulambda post = " <<  nulampost << std::endl;
    std::cout << "Value = " << sqrt((nulampost)/gen.chi_square()) << std::endl;
    *(ci.sigma) = sqrt((nulampost)/gen.chi_square());
    std::cout << "After --- *ci.sigma = " <<  *(ci.sigma) << std::endl;
    */

}


