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
   
   //control parameters
   size_t nd;
   size_t nummodels;
   size_t p;
   int tc;
   
   conf >> nd;
   conf >> nummodels;
   conf >> p;
   conf >> tc;

   // Read in arguments for each model
   std::vector<size_t> means_list;
   std::vector<size_t> m_list;
   std::vector<size_t> mh_list;

   double mean;
   size_t m;
   size_t mh;

   for(size_t i = 0;i<=nummodels;i++){
      conf >> mean;
      conf >> m;
      conf >> mh;
      means_list.push_back(mean);
      m_list.push_back(m);
      mh_list.push_back(mh);
   }

   // Get the design columns per emulator
   std::vector<std::vector<size_t>> x_cols_list(nummodels, std::vector<size_t>(1));
   std::vector<size_t> xcols, pvec;
   size_t ptemp, xcol;
   pvec.push_back(p); 
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


   //--------------------------------------------------
   //read in xp
   std::vector<double> xp;
   double xtemp;
   size_t np;
   std::stringstream xfss;
   std::string xfs;
   xfss << folder << xpcore << mpirank;
   xfs=xfss.str();
   std::ifstream xf(xfs);
   while(xf >> xtemp)
      xp.push_back(xtemp);
   np = xp.size()/pvec[0];
#ifndef SILENT
   cout << "node " << mpirank << " loaded " << np << " inputs of dimension " << p << " from " << xfs << endl;
#endif
   cout << "node " << mpirank << " loaded " << np << " inputs of dimension " << p << " from " << xfs << endl;

   //--------------------------------------------------
   //Construct prediction design matrices for each emulator -- essential when the emulators only use a subset of inputs from xp
   std::vector<std::vector<double>> xc_list(nummodels);

   size_t xcolsize = 0;
   xcol = 0;
   // Get the appropriate x columns
   for(size_t i=0;i<nummodels;i++){
      xcolsize = x_cols_list[i].size(); //x_cols_list is nummodel dimensional -- only for emulators
      for(size_t j=0;j<np;j++){
            for(size_t k=0;k<xcolsize;k++){
               xcol = x_cols_list[i][k] - 1;
               xc_list[i].push_back(xc_list[0][j*xcolsize + xcol]); //xc_list is nummodel dimensional -- only for emulators
            }
      }
      xc_list[i+1].insert(xc_list[i+1].end(),xc_list[i].begin(),xc_list[i].end()); 
   }

   //--------------------------------------------------
   //make xinfo
   std::vector<xinfo> xi_list(nummodels+1);
   std::vector<double> xivec;
   std::stringstream xifss;
   std::string xifs;
   std::ifstream xif;
   double xitemp;
   size_t indx = 0;

   for(int j=0;j<=nummodels;j++){
      xi_list[j].resize(pvec[j]);
      for(size_t i=0;i<pvec[j];i++) {
         // Get the next column in the x_cols_list -- important since emulators may have different inputs
         if(j>0){
               indx = (size_t)x_cols_list[j-1][i]; 
         }else{
               indx = i+1;
         }
         xifss << folder << xicore << (indx); 
         xifs=xifss.str();
         xif.open(xifs);
         while(xif >> xitemp){
            xivec.push_back(xitemp);
         }
         xi_list[j][i]=xivec;
         //Reset file strings
         xifss.str("");
         xif.close();
         xivec.clear();
      }
#ifndef SILENT
      cout << "&&& made xinfo\n";
#endif

    //summarize input variables:
#ifndef SILENT
      for(size_t i=0;i<p;i++){
         cout << "Variable " << i << " has numcuts=" << xi_list[j][i].size() << " : ";
         cout << xi_list[j][i][0] << " ... " << xi_list[j][i][xi_list[j][i].size()-1] << endl;
      }
#endif
    }


   //--------------------------------------------------
   //Set up model objects and MCMC
   //--------------------------------------------------    
   ambrt *ambm_list[nummodels]; //additive mean bart emulators
   psbrt *psbm_list[nummodels]; //product variance for bart emulators
   amxbrt axb(m_list[0]); // additive mean mixing bart
   psbrt pxb(mh_list[0]); //product model for mixing variance
   finfo fi;
   int l = 0; //Used for indexing emulators

   //finfo matrix
   if(mpirank > 0){
      fi = mxd::Ones(np, nummodels+1); //dummy initialize to matrix of 1's -- n0 x K+1 (1st column is discrepancy)
   }
   
   //Initialie the model mixing related objects
   axb.setxi(&xi_list[0]);   
   pxb.setxi(&xi_list[0]);
   axb.setfi(&fi, nummodels+1);

   // Initialize the emulators and associated variance models 
   for(size_t i=0;i<nummodels;i++){
      ambm_list[i] = new ambrt(m_list[i+1]);
      ambm_list[i]->setxi(&xi_list[i+1]);
      psbm_list[i] = new psbrt(mh_list[i+1]);
      psbm_list[i]->setxi(&xi_list[i+1]);
   }
   //--------------------------------------------
   // load files
   //--------------------------------------------
   // Initialize containers
   size_t temp=0;
   std::vector<std::vector<int>> e_ots(nummodels+1, std::vector<int>(temp));
   std::vector<std::vector<int>> e_oid(nummodels+1, std::vector<int>(temp));
   std::vector<std::vector<int>> e_ovar(nummodels+1, std::vector<int>(temp));
   std::vector<std::vector<int>> e_oc(nummodels+1, std::vector<int>(temp));
   std::vector<std::vector<double>> e_otheta(nummodels+1, std::vector<double>(temp));
   std::vector<std::vector<int>> e_sts(nummodels+1, std::vector<int>(temp));
   std::vector<std::vector<int>> e_sid(nummodels+1, std::vector<int>(temp));
   std::vector<std::vector<int>> e_svar(nummodels+1, std::vector<int>(temp));
   std::vector<std::vector<int>> e_sc(nummodels+1, std::vector<int>(temp));
   std::vector<std::vector<double>> e_stheta(nummodels+1, std::vector<double>(temp));
   std::vector<int> e_sstart(nummodels+1,0);
   size_t ind,im,imh;
   std::ifstream imf;
#ifndef SILENT
   if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif
   for(size_t j=0;j<=nummodels;j++){
      if(j == 0){
         imf.open(folder + modelname + ".fitmix");
      }else{
         imf.open(folder + modelname + ".fitemulate");
      }
      imf >> ind;
      imf >> im;
      imf >> imh;
   #ifdef _OPENMPI
      if(nd!=ind) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
      if(m!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
      if(mh!=imh) { cout << "Error loading posterior trees"  << endl; MPI_Finalize(); return 0; }
   #else
      if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
      if(m!=im) { cout << "Error loading posterior trees" << endl; return 0; }
      if(mh!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
   #endif

      // Model Mixing object
      imf >> temp;
      e_ots[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> e_ots[j].at(i);

      imf >> temp;
      e_oid[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> e_oid[j].at(i);

      imf >> temp;
      e_ovar[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> e_ovar[j].at(i);

      imf >> temp;
      e_oc[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> e_oc[j].at(i);

      imf >> temp;
      e_otheta[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_otheta[j].at(i);

      imf >> temp;
      e_sts[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> e_sts[j].at(i);

      imf >> temp;
      e_sid[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> e_sid[j].at(i);

      imf >> temp;
      e_svar[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> e_svar[j].at(i);

      imf >> temp;
      e_sc[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> e_sc[j].at(i);

      imf >> temp;
      e_stheta[j].resize(temp);
      for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_stheta[j].at(i);

      // close file if reading in model mixing data
      if(j == 0){imf.close();}
   
   }
   // close file for emulation data
   imf.close();

   //-------------------------------------------------
   // Setup containers for predictions 
   //-------------------------------------------------
   std::vector<std::vector<std::vector<double>>> tedraw_list(nummodels+1, std::vector<std::vector<double>>(nd, std::vector<double>(np)));
   std::vector<std::vector<std::vector<double>>> tedrawh_list(nummodels+1, std::vector<std::vector<double>>(nd, std::vector<double>(np)));
   std::vector<double*> fp_list(nummodels+1);
   std::vector<dinfo> dip_list(nummodels+1);
   for(size_t j=0;j<=nummodels;j++){
      fp_list[j] = new double[np];
      dip_list[j].y=fp_list[j]; dip_list[j].p = p; dip_list[j].n=np; dip_list[j].tc=1;
      if(j == 0){
         dip_list[j].x = &xp[0]; //mixing inputs
      }else{
         dip_list[j].x = &xc_list[j-1][0]; //emulator specific inputs
      }  
   }
   
   // Temporary vectors used for loading one model realization at a time.
   std::vector<int> onn(1,1);
   std::vector<std::vector<int> > oid(1, std::vector<int>(1));
   std::vector<std::vector<int> > ov(1, std::vector<int>(1));
   std::vector<std::vector<int> > oc(1, std::vector<int>(1));
   std::vector<std::vector<double> > otheta(1, std::vector<double>(1));
   std::vector<int> snn(1,1);
   std::vector<std::vector<int> > sid(1, std::vector<int>(1));
   std::vector<std::vector<int> > sv(1, std::vector<int>(1));
   std::vector<std::vector<int> > sc(1, std::vector<int>(1));
   std::vector<std::vector<double> > stheta(1, std::vector<double>(1));

   // Draw realizations of the posterior predictive.
   size_t curdx=0;
   size_t cumdx=0;
   size_t nt=1; //number of thetas
#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
#endif

   //-------------------------------------------------
   // Get predictions 
   //-------------------------------------------------
   // Mean trees first
   if(mpirank==0) cout << "Drawing mean response from posterior predictive" << endl;
   for(size_t k=0;k<=nummodels;k++){
      // Reset values
      m = m_list[k];
      cumdx = 0;
      curdx = 0;
      if(k==0){nt = nummodels+1;}else{nt = 1;}
      
      // Reset sizes of containers
      onn.resize(m,1);
      oid.resize(m, std::vector<int>(1));
      ov.resize(m, std::vector<int>(1));
      oc.resize(m, std::vector<int>(1));
      otheta.resize(m, std::vector<double>(1));

      for(size_t i=0;i<nd;i++) {
         curdx=0;
         for(size_t j=0;j<m;j++) {
            onn[j]=e_ots[k].at(i*m+j);
            oid[j].resize(onn[j]);
            ov[j].resize(onn[j]);
            oc[j].resize(onn[j]);
            otheta[j].resize(onn[j]*nt);
            // Loop through the nodes in a given tree
            for(size_t l=0;l<(size_t)onn[j];l++) {
               oid[j][l]=e_oid[k].at(cumdx+curdx+l);
               ov[j][l]=e_ovar[k].at(cumdx+curdx+l);
               oc[j][l]=e_oc[k].at(cumdx+curdx+l);
               if(k==0){
                  // Vector theta -- model mixing
                  for(size_t r=0;r<nt;r++){otheta[j][l*nt+r]=e_otheta[k].at((cumdx+curdx+l)*nt+r);}
               }else{
                  // Scalar theta -- emulators
                  otheta[j][l]=e_otheta[k].at(cumdx+curdx+l);
               }
            }
            curdx+=(size_t)onn[j];
         }
         cumdx+=curdx;

         // Load tree and draw relization
         if(k == 0){
            axb.loadtree_vec(0,m,onn,oid,ov,oc,otheta);
            axb.predict_mix(&dip_list[k],&fi);
         }else{
            ambm_list[k-1]->loadtree(0,m,onn,oid,ov,oc,otheta);
            ambm_list[k-1]->predict(&dip_list[k]);
            // Set prediction and update finfo
            for(size_t j=0;j<np;j++){
               tedraw_list[k][i][j] = fp_list[k][j] + means_list[k];
               fi(j,k) = tedraw_list[k][i][j];
            } 
         }
         
      }
   }
   
   // Variance trees second
   if(mpirank==0) cout << "Drawing sd response from posterior predictive" << endl;
   for(size_t k=0;k<=nummodels;k++){
      // Set values
      mh = mh_list[k];
      cumdx=0;
      curdx=0;

      // Reset sizes of containers
      snn.resize(mh,1);
      sid.resize(mh, std::vector<int>(1));
      sv.resize(mh, std::vector<int>(1));
      sc.resize(mh, std::vector<int>(1));
      stheta.resize(mh, std::vector<double>(1));
      
      for(size_t i=0;i<nd;i++) {
         curdx=0;
         for(size_t j=0;j<mh;j++) {
            snn[j]=e_sts[k].at(i*mh+j);
            sid[j].resize(snn[j]);
            sv[j].resize(snn[j]);
            sc[j].resize(snn[j]);
            stheta[j].resize(snn[j]);
            for(size_t k=0;k< (size_t)snn[j];k++) {
               sid[j][k]=e_sid[k].at(cumdx+curdx+k);
               sv[j][k]=e_svar[k].at(cumdx+curdx+k);
               sc[j][k]=e_sc[k].at(cumdx+curdx+k);
               stheta[j][k]=e_stheta[k].at(cumdx+curdx+k);
            }
            curdx+=(size_t)snn[j];
         }
         cumdx+=curdx;

         if(k == 0){
            // load tree and draw realization -- mixing variance
            pxb.loadtree(0,mh,snn,sid,sv,sc,stheta);
            pxb.predict(&dip_list[k]);
            for(size_t j=0;j<np;j++) tedrawh_list[k][i][j] = fp_list[k][j];
         }else{
            // load tree and draw realization -- emulator variance
            psbm_list[k-1]->loadtree(0,mh,snn,sid,sv,sc,stheta);
            psbm_list[k-1]->predict(&dip_list[k]);
            for(size_t j=0;j<np;j++) tedrawh_list[k][i][j] = fp_list[k][j];
         }
         
      }
   }

   //-------------------------------------------------
   // Save predictions to files 
   //-------------------------------------------------
#ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Posterior predictive draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif
   // Save the draws.
   if(mpirank==0) cout << "Saving posterior predictive draws...";
   std::ofstream omf(folder + modelname + ".mdraws" + std::to_string(mpirank));
   for(size_t k=0;k<=nummodels;k++){
      for(size_t i=0;i<nd;i++) {
         for(size_t j=0;j<np;j++)
            omf << std::scientific << tedraw_list[k][i][j] << " ";
         omf << endl;
      }
   }
   omf.close();

   std::ofstream smf(folder + modelname + ".sdraws" + std::to_string(mpirank));
   for(size_t k=0;k<=nummodels;k++){
      for(size_t i=0;i<nd;i++) {
         for(size_t j=0;j<np;j++)
            smf << std::scientific << tedrawh_list[k][i][j] << " ";
         smf << endl;
      }
   }
   smf.close();

   if(mpirank==0) cout << " done." << endl;

   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   MPI_Finalize();
#endif
   return 0;

}