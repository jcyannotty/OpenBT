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
#include "tree.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mbrt.h"
#include "ambrt.h"
#include "psbrt.h"
#include "tnorm.h"
#include "mxbrt.h"
#include "amxbrt.h"

using std::cout;
using std::endl;

#define MODEL_BT 1
#define MODEL_BINOMIAL 2
#define MODEL_POISSON 3
#define MODEL_BART 4
#define MODEL_HBART 5
#define MODEL_PROBIT 6
#define MODEL_MODIFIEDPROBIT 7
#define MODEL_MIXBART 9 //Skipped 8 because MERCK is 8 in cli.cpp
#define MODEL_MIXEMULATE 10

// Draw predictive realizations at the prediciton points, xp.
int main(int argc, char* argv[])
{
    std::string folder("");

    if(argc>1)
    {
        //argument on the command line is path to config file.
        folder=std::string(argv[1]);
        folder=folder+"/";
    }

    //-----------------------------------------------------------
    //random number generation -- only used in model mixing with function discrepancy right now
    crn gen;
    gen.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()));

    //--------------------------------------------------
    //process args
    std::ifstream conf(folder+"config.pred");
    std::string modelname;
    int modeltype;
    std::string xicore;
    std::string xpcore;

    //model name, xi and xp
    conf >> modelname;
    conf >> modeltype;
    conf >> xicore;
    conf >> xpcore;
    
    //number of saved draws and number of trees
    size_t nd;
    size_t m;
    size_t mh;

    conf >> nd;
    conf >> m;
    conf >> mh;

    //number of predictors
    size_t p, nummodels;
    conf >> p;
    conf >> nummodels;

    //thread count
    int tc;
    conf >> tc;

    //data means
    std::vector<double> means_list;
    double means;

    // Data means
    for(size_t i=0;i<=nummodels;i++){
            conf >> means;
            means_list.push_back(means);
    }

    // Get the design columns per emulator
    std::vector<std::vector<size_t>> x_cols_list(nummodels, std::vector<size_t>(1));
    std::vector<size_t> xcols, pvec;
    size_t ptemp, xcol;
    for(int i=0;i<nummodels;i++){
        conf >> ptemp;
        pvec.push_back(ptemp);
        x_cols_list[i].resize(ptemp);
        for(size_t j = 0; j<ptemp; j++){
            conf >> xcol;
            x_cols_list[i][j] = xcol;
        }

    }
   
   conf.close();
   
   //MPI initialization
   int mpirank=0;
#ifdef _OPENMPI
   int mpitc;
   MPI_Init(NULL,NULL);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
   MPI_Comm_size(MPI_COMM_WORLD,&mpitc);
#ifndef SILENT
   cout << "\nMPI: node " << mpirank << " of " << mpitc << " processes." << endl;
#endif
   if(tc<=1){
      cout << "Error: tc=" << tc << endl;
      MPI_Finalize();
      return 0; //need at least 2 processes! 
   } 
   if(tc!=mpitc) {
      cout << "Error: tc does not match mpitc" << endl;
      MPI_Finalize();
      return 0; //mismatch between how MPI was started and how the data is prepared according to tc.
   }
// #else
//    if(tc!=1) return 0; //serial mode should have no slave threads!
#endif


   //--------------------------------------------------
   // Banner
   if(mpirank==0) {
      cout << endl;
      cout << "-----------------------------------" << endl;
      cout << "OpenBT mixing prediction interface" << endl;
      cout << "Loading config file at " << folder << endl;
   }

}