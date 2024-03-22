/*
Name: Command Line Interface for BART Bayesian Model Mixing - Two Step Process
Auth: JCY (yannotty.1@buckeyemail.osu.edu)
Desc: Trains the BMM model assuming a two step process. This means the simulator predictions
    are read in via a text file and NO emulation or calibrtion is required.
*/

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
#include "psbrt.h"
#include "tnorm.h"
#include "mxbrt.h"
#include "amxbrt.h"

using std::cout;
using std::endl;

#define MODEL_BARTBMMM 10

int main(int argc, char* argv[])
{
   std::string folder("");

   if(argc>1)
   {
      std::string confopt("--conf");
      if(confopt.compare(argv[1])==0) {
#ifdef _OPENMPI
         return 101;
#else
         return 100;
#endif
      }

      //otherwise argument on the command line is path to conifg file.
      folder=std::string(argv[1]);
      folder=folder+"/";
   }


   //-----------------------------------------------------------
   //random number generation
   crn gen;
   gen.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                   .time_since_epoch()
                                   .count()));

   //--------------------------------------------------
   //process args
   std::ifstream conf(folder+"config");

   // model type
   int modeltype;
   conf >> modeltype;

   // core filenames for x,y,f input --- f input is for model mixing
   std::string xcore,ycore;
   conf >> xcore;
   conf >> ycore;
   
   //number of trees
   size_t m;
   size_t mh;
   conf >> m;
   conf >> mh;

   //nd and burn
   size_t nd;
   size_t burn;
   size_t nadapt;
   size_t adaptevery;
   conf >> nd;
   conf >> burn;
   conf >> nadapt;
   conf >> adaptevery;

   //mu prior (tau, ambrt --OR-- beta0,tau, amxbrt) and sigma prior (lambda,nu, psbrt) 
   double tau;
   double beta0;
   double overalllambda;
   double overallnu;
   conf >> tau;
   conf >> beta0;
   conf >> overalllambda;
   conf >> overallnu;
   
   //tree prior
   double alpha;
   double mybeta;
   double alphah;
   double mybetah;
   size_t maxd;
   conf >> alpha;
   conf >> mybeta;
   conf >> alphah;
   conf >> mybetah;
   conf >> maxd;

   //thread count
   int tc;
   conf >> tc;

   //sigma vector
   std::string score;
   conf >> score;

   //change variable
   std::string chgvcore;
   conf >> chgvcore;

   //fhat and sdhat for model mixing 
   std::string fcore, fsdcore;
   conf >> fcore;
   conf >> fsdcore;

   //non-stationary prior for mixing True/False - read in as a string (reading is as a bool prevented the rest of the values from being read, not sure why)
   std::string eftprior_str;
   bool eftprior = false;
   conf >> eftprior_str;
   if(eftprior_str == "TRUE" || eftprior_str == "True"){ eftprior = true; }
   
   //selection prior
   //std::string selectp_str;
   //bool selectp = false;
   //conf >> selectp_str;
   //if(selectp_str == "TRUE" || selectp_str == "True"){ selectp = true; }

   // Random path arguments
   std::string randpath_str;
   std::string modbd_str;
   double gam, q, sh1, sh2;
   bool randpath = false;
   bool modbd = false;
   conf >> randpath_str;
   conf >> gam;
   conf >> q;
   conf >> sh1;
   conf >> sh2;
   conf >> modbd_str;
   if(randpath_str == "TRUE" || randpath_str == "True"){ randpath = true; }
   if(modbd_str == "TRUE" || modbd_str == "True"){ modbd = true; }

   //control
   double pbd;
   double pb;
   double pbdh;
   double pbh;
   double stepwpert;
   double stepwperth;
   double probchv;
   double probchvh;
   size_t minnumbot;
   size_t minnumboth;
   size_t printevery;
   size_t writebatchsz;
   std::string xicore;
   std::string modelname;
   conf >> pbd;
   conf >> pb;
   conf >> pbdh;
   conf >> pbh;
   conf >> stepwpert;
   conf >> stepwperth;
   conf >> probchv;
   conf >> probchvh;
   conf >> minnumbot;
   conf >> minnumboth;
   conf >> printevery;
   conf >> writebatchsz;
   conf >> xicore;
   conf >> modelname;

   bool dopert=true;
   bool doperth=true;
   if(probchv<0) dopert=false;
   if(probchvh<0) doperth=false;
   
   //summary statistics yes/no
   bool summarystats = false;
   std::string summarystats_str;
   conf >> summarystats_str;
   if(summarystats_str=="TRUE"){ summarystats = true; }
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
   if(tc<=1) return 0; //need at least 2 processes!
   if(tc!=mpitc) return 0; //mismatch between how MPI was started and how the data is prepared according to tc.
// #else
//    if(tc!=1) return 0; //serial mode should have no slave threads!
#endif


   //--------------------------------------------------
   // Banner
   if(mpirank==0) {
      cout << endl;
      cout << "-----------------------------------" << endl;
      cout << "OpenBT Model Mixing Two Step Process Command-Line Interface" << endl;
      cout << "Loading config file at " << folder << endl;
   }

   //--------------------------------------------------
   //read in y 
   std::vector<double> y;
   double ytemp;
   size_t n=0;
#ifdef _OPENMPI
   if(mpirank>0) { //only load data on slaves
#endif
      std::stringstream yfss;
      std::string yfs;
      yfss << folder << ycore << mpirank;
      yfs=yfss.str();
      std::ifstream yf(yfs);
      while(yf >> ytemp)
         y.push_back(ytemp);
      n=y.size();
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << n << " from " << yfs <<endl;
#endif
#ifdef _OPENMPI
   }
#endif

   //--------------------------------------------------
   //read in x 
   std::vector<double> x;
   double xtemp;
   size_t p=0;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      std::stringstream xfss;
      std::string xfs;
      xfss << folder << xcore << mpirank;
      xfs=xfss.str();
      std::ifstream xf(xfs);
      while(xf >> xtemp)
         x.push_back(xtemp);
      p = x.size()/n;
      cout << "p = " << p << endl;
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << p << " from " << xfs << endl;
#endif
#ifdef _OPENMPI
   }
   int tempp = (unsigned int) p;
   MPI_Allreduce(MPI_IN_PLACE,&tempp,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   if(mpirank>0 && p != ((size_t) tempp)) { cout << "PROBLEM LOADING DATA" << endl; MPI_Finalize(); return 0;}
   p=(size_t)tempp;
#endif

   //--------------------------------------------------
   //Initialize f matrix and make finfo -- used only for model mixing 
   std::vector<double> f;
   double ftemp;
   size_t k=0;
   finfo fi;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      std::stringstream ffss;
      std::string ffs;
      ffss << folder << fcore << mpirank;
      ffs=ffss.str();
      std::ifstream ff(ffs);
      while(ff >> ftemp)
         f.push_back(ftemp);
      k = f.size()/n;
      
      //Make finfo on the slave node
      makefinfo(k,n,&f[0],fi);
      cout << "node " << mpirank << " loaded " << n << " mixing inputs of dimension " << k << " from " << ffs << endl;
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << n << " mixing inputs of dimension " << k << " from " << ffs << endl;
#endif
#ifdef _OPENMPI
   }
   int tempk = (unsigned int) k;
   MPI_Allreduce(MPI_IN_PLACE,&tempk,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   if(mpirank>0 && k != ((size_t) tempk)) { cout << "PROBLEM LOADING DATA" << endl; MPI_Finalize(); return 0;}
   k=(size_t)tempk;
#endif

   //--------------------------------------------------
   //Initialize f std information -- used only for model mixing when fdiscrepancy = TRUE 
   std::vector<double> fsdvec;
   double fsdtemp;
   finfo fsd; 
   size_t ksd = 0;
   if(eftprior){   
   #ifdef _OPENMPI
      if(mpirank>0) {
   #endif      
         std::stringstream fsdfss;
         std::string fsdfs;
         fsdfss << folder << fsdcore << mpirank;
         fsdfs=fsdfss.str();
         std::ifstream fsdf(fsdfs);
         while(fsdf >> fsdtemp)
            fsdvec.push_back(fsdtemp);
         ksd = fsdvec.size()/n; 

         //Make finfo on the slave node
         makefinfo(k,n,&fsdvec[0],fsd);
   
   #ifndef SILENT
         //cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << kdm << " from " << fdmfs << endl;
         cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << ksd << " from " << fsdfs << endl;
   #endif
   #ifdef _OPENMPI
      }
      int tempkd = (unsigned int) k;
      MPI_Allreduce(MPI_IN_PLACE,&tempkd,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      //if(mpirank>0 && kdm != ((size_t) tempkd)) { cout << "PROBLEM LOADING DISCREPANCY DATA" << endl; MPI_Finalize(); return 0;}
      if(mpirank>0 && ksd != ((size_t) tempkd)) { cout << "PROBLEM LOADING DISCREPANCY DATA" << endl; MPI_Finalize(); return 0;}
   #endif   
   }   
   
   //--------------------------------------------------
   //dinfo
   dinfo di;
   di.n=0;di.p=p,di.x = NULL;di.y=NULL;di.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      di.n=n; di.x = &x[0]; di.y = &y[0]; 
#ifdef _OPENMPI
   }
#endif

   //--------------------------------------------------
   //read in sigmav  -- same as above.
   std::vector<double> sigmav;
   double stemp;
   size_t nsig=0;
#ifdef _OPENMPI
   if(mpirank>0) { //only load data on slaves
#endif
      std::stringstream sfss;
      std::string sfs;
      sfss << folder << score << mpirank;
      sfs=sfss.str();
      std::ifstream sf(sfs);
      while(sf >> stemp)
         sigmav.push_back(stemp);
      nsig=sigmav.size();
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << nsig << " from " << sfs <<endl;
#endif
#ifdef _OPENMPI
      if(n!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; MPI_Finalize(); return 0; }
   }
#else
   if(n!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; return 0; }
#endif

   double *sig=&sigmav[0];
   dinfo disig;
   disig.n=0; disig.p=p; disig.x=NULL; disig.y=NULL; disig.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      disig.n=n; disig.x=&x[0]; disig.y=sig; 
#ifdef _OPENMPI
   }
#endif

   //--------------------------------------------------
   // read in the initial change of variable rank correlation matrix
   std::vector<std::vector<double>> chgv;
   std::vector<double> cvvtemp;
   double cvtemp;
   std::ifstream chgvf(folder + chgvcore);
   for(size_t i=0;i<di.p;i++) {
      cvvtemp.clear();
      for(size_t j=0;j<di.p;j++) {
         chgvf >> cvtemp;
         cvvtemp.push_back(cvtemp);
      }
      chgv.push_back(cvvtemp);
   }
#ifndef SILENT
   cout << "mpirank=" << mpirank << ": change of variable rank correlation matrix loaded:" << endl;
#endif


   //--------------------------------------------------
   // decide what variables each slave node will update in change-of-variable proposals.
#ifdef _OPENMPI
   int* lwr=new int[tc];
   int* upr=new int[tc];
   lwr[0]=-1; upr[0]=-1;
   for(size_t i=1;i<(size_t)tc;i++) { 
      lwr[i]=-1; upr[i]=-1; 
      calcbegend(p,i-1,tc-1,&lwr[i],&upr[i]);
      if(p>1 && lwr[i]==0 && upr[i]==0) { lwr[i]=-1; upr[i]=-1; }
   }

#ifndef SILENT
   if(mpirank>0) cout << "Slave node " << mpirank << " will update variables " << lwr[mpirank] << " to " << upr[mpirank]-1 << endl;
#endif
#endif

   //--------------------------------------------------
   //make xinfo
   xinfo xi;
   xi.resize(p);

   for(size_t i=0;i<p;i++) {
      std::vector<double> xivec;
      double xitemp;

      std::stringstream xifss;
      std::string xifs;
      xifss << folder << xicore << (i+1);
      xifs=xifss.str();
      std::ifstream xif(xifs);
      while(xif >> xitemp)
         xivec.push_back(xitemp);
      xi[i]=xivec;
   }
#ifndef SILENT
   cout << "&&& made xinfo\n";
#endif

   //summarize input variables:
#ifndef SILENT
   for(size_t i=0;i<p;i++)
   {
      cout << "Variable " << i << " has numcuts=" << xi[i].size() << " : ";
      cout << xi[i][0] << " ... " << xi[i][xi[i].size()-1] << endl;
   }
#endif

   //Setup additive mean mixing bart model
   amxbrt axb(m);

   //cutpoints
   axb.setxi(&xi);    //set the cutpoints for this model object
   //function output information
   axb.setfi(&fi, k);
   //set individual function discrepacnies if provided 
   if(eftprior) {axb.setfsd(&fsd);}
   // Random path setter (must be before data right now...)
   if(randpath){
      axb.setrpi(gam,q,sh1,sh2,n,modbd);
   }
   //data objects
   axb.setdata_vec(&di);  //set the data
   //thread count
   axb.settc(tc-1);      //set the number of slaves when using MPI.
   //mpi rank
#ifdef _OPENMPI
   axb.setmpirank(mpirank);  //set the rank when using MPI.
   axb.setmpicvrange(lwr,upr); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   axb.settp(alpha, //the alpha parameter in the tree depth penalty prior
         mybeta     //the beta parameter in the tree depth penalty prior
         );
   axb.setmaxd(maxd); // set maxdepth
   //MCMC info
   axb.setmi(
         pbd,  //probability of birth/death
         pb,  //probability of birth
         minnumbot,    //minimum number of observations in a bottom node
         dopert, //do perturb/change variable proposal?
         stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
   
   axb.setci(tau,beta0,sig);
   mxd loss;
      

   //--------------------------------------------------
   //setup psbrt object
   psbrt psbm(mh,overalllambda);

   //make di for psbrt object
   dinfo dips;
   dips.n=0; dips.p=p; dips.x=NULL; dips.y=NULL; dips.tc=tc;
   double *r=NULL;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      r = new double[n];
      for(size_t i=0;i<n;i++) r[i]=sigmav[i];
      dips.x=&x[0]; dips.y=r; dips.n=n;
#ifdef _OPENMPI
   }
#endif

   double opm=1.0/((double)mh);
   double nu=2.0*pow(overallnu,opm)/(pow(overallnu,opm)-pow(overallnu-2.0,opm));
   double lambda=pow(overalllambda,opm);

   //cutpoints
   psbm.setxi(&xi);    //set the cutpoints for this model object
   //data objects
   psbm.setdata(&dips);  //set the data
   //thread count
   psbm.settc(tc-1); 
   //mpi rank
#ifdef _OPENMPI
   psbm.setmpirank(mpirank);  //set the rank when using MPI.
   psbm.setmpicvrange(lwr,upr); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   psbm.settp(alphah, //the alpha parameter in the tree depth penalty prior
         mybetah     //the beta parameter in the tree depth penalty prior
         );
   psbm.setmi(
         pbdh,  //probability of birth/death
         pbh,  //probability of birth
         minnumboth,    //minimum number of observations in a bottom node
         doperth, //do perturb/change variable proposal?
         stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
   psbm.setci(nu,lambda);


    //--------------------------------------------------
    //run mcmc
    std::vector<int> onn(writebatchsz*m,1);
    std::vector<std::vector<int> > oid(writebatchsz*m, std::vector<int>(1));
    std::vector<std::vector<int> > ovar(writebatchsz*m, std::vector<int>(1));
    std::vector<std::vector<int> > oc(writebatchsz*m, std::vector<int>(1));
    std::vector<std::vector<double> > otheta(writebatchsz*m, std::vector<double>(1));

    std::vector<double> ogam(nd*m, 1);
    //std::vector<double> osig(nd,1);

    std::vector<int> snn(nd*mh,1);
    std::vector<std::vector<int> > sid(nd*mh, std::vector<int>(1));
    std::vector<std::vector<int> > svar(nd*mh, std::vector<int>(1));
    std::vector<std::vector<int> > sc(nd*mh, std::vector<int>(1));
    std::vector<std::vector<double> > stheta(nd*mh, std::vector<double>(1));

    brtMethodWrapper faxb(&brt::f,axb);
    brtMethodWrapper fpsbm(&brt::f,psbm);

    // Init containers to write mean objects (other containers defined later, define up here bc of batch sampling)
    std::vector<int>* e_ots=new std::vector<int>(writebatchsz*m);
    std::vector<int>* e_oid=new std::vector<int>;
    std::vector<int>* e_ovar=new std::vector<int>;
    std::vector<int>* e_oc=new std::vector<int>;
    std::vector<double>* e_otheta=new std::vector<double>;


#ifdef _OPENMPI
    double tstart=0.0,tend=0.0;
    if(mpirank==0) tstart=MPI_Wtime();
    if(mpirank==0) cout << "Starting MCMC..." << endl;
#else
    cout << "Starting MCMC..." << endl;
#endif
    //------------------------------------------------------------
    // Adapt Stage
    //------------------------------------------------------------
    for(size_t i=0;i<nadapt;i++) { 
        if((i % printevery) ==0 && mpirank==0) cout << "Adapt iteration " << i << endl;
    #ifdef _OPENMPI
        if(mpirank==0){axb.drawvec(gen);} else {axb.drawvec_mpislave(gen);}
        if(randpath){ 
            if(mpirank==0){ axb.drawgamma(gen);}else {axb.drawgamma_mpi(gen);}
        }
    #else
        axb.drawvec(gen);
        if(randpath) axb.drawgamma(gen);
    #endif

        // Update the which are fed into resiudals the variance model
        dips = di;
        dips -= faxb;
        if((i+1)%adaptevery==0 && mpirank==0) axb.adapt();
        if((i+1)%adaptevery==0 && mpirank==0 && randpath) axb.rpath_adapt();
    #ifdef _OPENMPI
        if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
    #else
        psbm.draw(gen);
    #endif
        disig = fpsbm;
        if((i+1)%adaptevery==0 && mpirank==0) psbm.adapt();
    
    }

   //------------------------------------------------------------
   // Enter the burn-in stage
   //------------------------------------------------------------    
   for(size_t i=0;i<burn;i++) {
      if((i % printevery) ==0 && mpirank==0) cout << "Burn iteration " << i << endl;
#ifdef _OPENMPI
      if(mpirank==0){ axb.drawvec(gen);}else {axb.drawvec_mpislave(gen);}
      if(randpath){ if(mpirank==0){ axb.drawgamma(gen);}else {axb.drawgamma_mpi(gen);}}
#else
      axb.drawvec(gen);
      if(randpath) axb.drawgamma(gen);
#endif
      // Update the which are fed into resiudals the variance model
      dips = di;
      dips -= faxb;
      // Draw sigma
#ifdef _OPENMPI
      if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
#else
      psbm.draw(gen);
#endif
      disig = fpsbm; 
   }

    //------------------------------------------------------------
    // Enter the Draw stage
    //------------------------------------------------------------    
    size_t batchnum = 0; //batch number for writing the results
    size_t bnd; // number of draws per batch
    if(summarystats) {
        axb.setstats(true);
        psbm.setstats(true);
    }
    for(size_t i=0;i<nd;i++) {
        if((i % printevery) ==0 && mpirank==0) cout << "Draw iteration " << i << endl;
#ifdef _OPENMPI
    if(mpirank==0){axb.drawvec(gen); }else{ axb.drawvec_mpislave(gen);}
    if(randpath){ if(mpirank==0){ axb.drawgamma(gen);}else {axb.drawgamma_mpi(gen);}}
      
#else
    axb.drawvec(gen);
    if(randpath) axb.drawgamma(gen);
#endif
    dips = di;
    dips -= faxb;
#ifdef _OPENMPI
    if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
#else
    psbm.draw(gen);
#endif
    disig = fpsbm;
    //if(mpirank ==1 && (i%printevery)==0){cout << axb.getsigma() << endl;}

    //save tree to vec format
    if(mpirank==0) {
        // Save vectors
        size_t bind = i - batchnum*writebatchsz;        
        axb.savetree_vec(bind,m,onn,oid,ovar,oc,otheta); 
        psbm.savetree(i,mh,snn,sid,svar,sc,stheta); 

        if(randpath){
            std::vector<double> tempgam(m,0);
            tempgam = axb.getgamma();
            for(size_t j=0;j<m;j++){ogam.at(i*m+j) = tempgam[j];}
        }

        // Write the mean tree results in batches....
        if(((i+1) % writebatchsz) == 0 || i == (nd-1)){
            bnd = (i+1) - writebatchsz*batchnum;
            e_ots->resize(bnd*m);
            for(size_t i=0;i<bnd;i++){
                for(size_t j=0;j<m;j++) {
                    e_ots->at(i*m+j)=static_cast<int>(oid[i*m+j].size());
                    e_oid->insert(e_oid->end(),oid[i*m+j].begin(),oid[i*m+j].end());
                    e_ovar->insert(e_ovar->end(),ovar[i*m+j].begin(),ovar[i*m+j].end());
                    e_oc->insert(e_oc->end(),oc[i*m+j].begin(),oc[i*m+j].end());
                    e_otheta->insert(e_otheta->end(),otheta[i*m+j].begin(),otheta[i*m+j].end());
                }
            }
            
            //write out to file
            std::ofstream omf;
            if(batchnum == 0){
                // Create the file 
                omf.open(folder + modelname + ".fit");
            }else{
                // Append to the existing file
                omf.open(folder + modelname + ".fit", std::ios_base::app);
            }
                        
            omf << bnd << endl;
            omf << m << endl;
            omf << e_ots->size() << endl;
            for(size_t i=0;i<e_ots->size();i++) omf << e_ots->at(i) << endl;
            omf << e_oid->size() << endl;
            for(size_t i=0;i<e_oid->size();i++) omf << e_oid->at(i) << endl;
            omf << e_ovar->size() << endl;
            for(size_t i=0;i<e_ovar->size();i++) omf << e_ovar->at(i) << endl;
            omf << e_oc->size() << endl;
            for(size_t i=0;i<e_oc->size();i++) omf << e_oc->at(i) << endl;
            omf << e_otheta->size() << endl;
            for(size_t i=0;i<e_otheta->size();i++) omf << std::scientific << e_otheta->at(i) << endl;
            omf.close();
            
            // Increase batch number
            batchnum += 1;

            // Clear the vectors for writing
            e_ots->clear();
            e_oid->clear();
            e_ovar->clear();
            e_oc->clear();
            e_otheta->clear();

            // Clear and Resize vectors for storing
            onn.clear();
            oid.clear();
            ovar.clear();
            oc.clear();
            otheta.clear();

            onn.resize(writebatchsz*m,1);
            oid.resize(writebatchsz*m, std::vector<int>(1));
            ovar.resize(writebatchsz*m, std::vector<int>(1));
            oc.resize(writebatchsz*m, std::vector<int>(1));
            otheta.resize(writebatchsz*m, std::vector<double>(1));
        }
    }
}

#ifdef _OPENMPI
    if(mpirank==0) {
        tend=MPI_Wtime();
        cout << "Training time was " << (tend-tstart)/60.0 << " minutes." << endl;
    }
#endif

    //Flatten posterior trees to a few (very long) vectors so we can just pass pointers
    //to these vectors back to R (which is much much faster than copying all the data back).
    if(mpirank==0) {
        cout << "Returning posterior, please wait...";
        std::vector<int>* e_sts=new std::vector<int>(nd*mh);
        std::vector<int>* e_sid=new std::vector<int>;
        std::vector<int>* e_svar=new std::vector<int>;
        std::vector<int>* e_sc=new std::vector<int>;
        std::vector<double>* e_stheta=new std::vector<double>;
        for(size_t i=0;i<nd;i++){
            for(size_t j=0;j<mh;j++) {
            e_sts->at(i*mh+j)=static_cast<int>(sid[i*mh+j].size());
            e_sid->insert(e_sid->end(),sid[i*mh+j].begin(),sid[i*mh+j].end());
            e_svar->insert(e_svar->end(),svar[i*mh+j].begin(),svar[i*mh+j].end());
            e_sc->insert(e_sc->end(),sc[i*mh+j].begin(),sc[i*mh+j].end());
            e_stheta->insert(e_stheta->end(),stheta[i*mh+j].begin(),stheta[i*mh+j].end());
            }
        }
        std::ofstream smf(folder + modelname + ".sfit");
        smf << nd << endl;
        smf << mh << endl;
        smf << e_sts->size() << endl;
        for(size_t i=0;i<e_sts->size();i++) smf << e_sts->at(i) << endl;
        smf << e_sid->size() << endl;
        for(size_t i=0;i<e_sid->size();i++) smf << e_sid->at(i) << endl;
        smf << e_svar->size() << endl;
        for(size_t i=0;i<e_svar->size();i++) smf << e_svar->at(i) << endl;
        smf << e_sc->size() << endl;
        for(size_t i=0;i<e_sc->size();i++) smf << e_sc->at(i) << endl;
        smf << e_stheta->size() << endl;
        for(size_t i=0;i<e_stheta->size();i++) smf << std::scientific << e_stheta->at(i) << endl;
        smf.close();

        std::ofstream ogf(folder + modelname + ".rpg"); // random path gamma
        if(randpath){
            for(size_t i=0;i<nd;i++){
            for(size_t j=0;j<m;j++) {ogf << ogam.at(i*m+j) << endl;}
            } 
            ogf.close();
        }

   }
   
   // summary statistics
   if(summarystats) {
      cout << "Calculating summary statistics" << endl;
      unsigned int varcount[p];
      for(size_t i=0;i<p;i++) varcount[i]=0;
      unsigned int tmaxd=0;
      unsigned int tmind=0;
      double tavgd=0.0;

      axb.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
      tavgd/=(double)(nd*m);
      cout << "Average tree depth (amxbrt): " << tavgd << endl;
      cout << "Maximum tree depth (amxbrt): " << tmaxd << endl;
      cout << "Minimum tree depth (amxbrt): " << tmind << endl;
      cout << "Vartivity summary (amxbrt)" << endl;
      for(size_t i=0;i<p;i++)
         cout << "Var " << i << ": " << varcount[i] << endl;

   }
   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   delete[] lwr;
   delete[] upr;
   MPI_Finalize();
#endif
   return 0;
}
