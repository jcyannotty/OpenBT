
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
#include "mcbrt.h"
#include "amcbrt.h"
#include "sbrt.h"
#include "psbrt.h"
#include "parameters.h"

using std::cout;
using std::endl;

#define MODEL_OSBART 1

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
    std::ifstream conf(folder+"config.calibratepred");
    std::string modelname;
    int modeltype;
    std::string xicore;
    std::string xpcore;
    std::string fpcore;

    //model name, xi and xp
    conf >> modelname;
    conf >> modeltype;
    conf >> xicore;
    conf >> xpcore;
    conf >> fpcore;

    //number of saved draws and number of trees
    size_t nd;
    size_t m;
    size_t mh;

    conf >> nd;
    conf >> m;
    conf >> mh;

    //number of predictors
    size_t p,pu,px;
    conf >> p;
    conf >> pu;
    px = p-pu;

    //thread count
    int tc;
    conf >> tc;

    //mean offset
    double yfmean,ycmean;
    conf >> yfmean;
    conf >> ycmean;

    std::vector<size_t> ucols;  
    size_t tempucol;
    for(size_t i=0;i<pu;i++){
        conf >> tempucol;
        ucols.push_back(tempucol);
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
#endif


   //--------------------------------------------------
   // Banner
   if(mpirank==0) {
      cout << endl;
      cout << "-----------------------------------" << endl;
      cout << "OpenBT Calibration prediction CLI" << endl;
      cout << "Loading config file at " << folder << endl;
   }

   //--------------------------------------------------
   //read in xp.
   std::vector<double> xp;
   double xtemp;
   size_t np;

   std::stringstream xfss;
   std::string xfs;
   xfss << folder << xpcore << mpirank;
   xfs=xfss.str();
   std::ifstream xfile(xfs);
   while(xfile >> xtemp)
      xp.push_back(xtemp);
   np = xp.size()/p;
#ifndef SILENT
   cout << "node " << mpirank << " loaded " << np << " inputs of dimension " << p << " from " << xfs << endl;
#endif
   cout << "node " << mpirank << " loaded " << np << " inputs of dimension " << p << " from " << xfs << endl;
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
   if(mpirank==0)
      for(size_t i=0;i<p;i++)
      {
         cout << "Variable " << i << " has numcuts=" << xi[i].size() << " : ";
         cout << xi[i][0] << " ... " << xi[i][xi[i].size()-1] << endl;
      }
#endif
   //--------------------------------------------------
   //Initialize f matrix and make finfo
    std::vector<double> fpd;
    double ftemp;
    finfo fi_test(np,2);
    std::stringstream ffss;
    std::string ffs;
    ffss << folder << fpcore << mpirank; //get the file to read in -- every processor reads in a different file (dictated by mpirank and R files)
    ffs=ffss.str();
    std::ifstream ff(ffs);
    while(ff >> ftemp)
        fpd.push_back(ftemp);
    Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> fi_test_temp(fpd.data(),2,np);
    fi_test = fi_test_temp.transpose();

#ifndef SILENT
   cout << "&&& made finfo for test data\n";
#endif
    //--------------------------------------------------
    // Get nf and nc split
    size_t nf,nc;
    nf = fi_test.colwise().sum()(1);
    nc = np-nf;

    // Split the data into xf and xc
    std::vector<double> xf, xc, xftemp;
    for(size_t i=0;i<np;i++){
        xftemp.clear();
        xftemp = {xp.begin()+p*i,xp.begin()+p*(i+1)};
        if(fi_test(i,1)==1){
            xf.insert(xf.end(),xftemp.begin(),xftemp.end());
        }else{
            xc.insert(xc.end(),xftemp.begin(),xftemp.end());
        }
    }

    //--------------------------------------------------
    // set up amcbrt object
    amcbrt acb(m);
    acb.setxi(&xi); //set the cutpoints for this model object
    acb.setfi(&fi_test, 2); //set the function output for this model object -- main use is to set k 
    //if(fdiscrepancy) {axb.setfdelta(&fdeltamean, &fdeltasd);}  //set individual function discrepacnies if provided -- main use is to set fdiscrepancy to TRUE
    
    //setup psbrt object
    psbrt psbmf(mh);
    psbmf.setxi(&xi); //set the cutpoints for this model object

    //setup psbrt object
    psbrt psbmc(mh);
    psbmc.setxi(&xi); //set the cutpoints for this model object
    
    //setup calibration parameter object
    param uvec(pu);

    //load from file
#ifndef SILENT
    if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif
    size_t ind,im,imh;
    std::ifstream imf(folder + modelname + ".fit");
    imf >> ind;
    imf >> im;
    imf >> imh;
#ifdef _OPENMPI
    if(nd!=ind) { cout << "Error loading posterior trees"<< "nd = " << nd << " -- ind = " << ind << endl; MPI_Finalize(); return 0; }
    if(m!=im) { cout << "Error loading posterior trees" << "m = " << m << " -- im = " << im<< endl; MPI_Finalize(); return 0; }
    if(mh!=imh) { cout << "Error loading posterior trees"  << endl; MPI_Finalize(); return 0; }
#else
    if(nd!=ind) { cout << "Error loading posterior trees" << "nd = " << nd << " -- ind = " << ind << endl; return 0; }
    if(m!=im) { cout << "Error loading posterior trees" << "m = " << m << " -- im = " << im<< endl; return 0; }
    if(mh!=imh) { cout << "Error loading posterior trees"  << endl; return 0; }
#endif
    // Mean tree information
    size_t temp=0;
    imf >> temp;
    std::vector<int> e_ots(temp);
    for(size_t i=0;i<temp;i++) imf >> e_ots.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_oid(temp);
    for(size_t i=0;i<temp;i++) imf >> e_oid.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_ovar(temp);
    for(size_t i=0;i<temp;i++) imf >> e_ovar.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_oc(temp);
    for(size_t i=0;i<temp;i++) imf >> e_oc.at(i);

    temp=0;
    imf >> temp;
    std::vector<double> e_otheta(temp);
    for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_otheta.at(i);

    // Product variance trees for field data
    temp=0;
    imf >> temp;
    std::vector<int> e_sfts(temp);
    for(size_t i=0;i<temp;i++) imf >> e_sfts.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_sfid(temp);
    for(size_t i=0;i<temp;i++) imf >> e_sfid.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_sfvar(temp);
    for(size_t i=0;i<temp;i++) imf >> e_sfvar.at(i);

    temp=0;
    imf >> temp;
    std::vector<int> e_sfc(temp);
    for(size_t i=0;i<temp;i++) imf >> e_sfc.at(i);

    temp=0;
    imf >> temp;
    std::vector<double> e_sftheta(temp);
    for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_sftheta.at(i);

    std::vector<int> e_scts;
    std::vector<int> e_scid;
    std::vector<int> e_scvar;
    std::vector<int> e_scc;
    std::vector<double> e_sctheta;
    if(nc>0){
        // Product variance trees for model runs
        temp=0;
        imf >> temp;
        e_scts.resize(temp);
        for(size_t i=0;i<temp;i++) imf >> e_scts.at(i);

        temp=0;
        imf >> temp;
        e_scid.resize(temp);
        for(size_t i=0;i<temp;i++) imf >> e_scid.at(i);

        temp=0;
        imf >> temp;
        e_scvar.resize(temp);
        for(size_t i=0;i<temp;i++) imf >> e_scvar.at(i);

        temp=0;
        imf >> temp;
        e_scc.resize(temp);
        for(size_t i=0;i<temp;i++) imf >> e_scc.at(i);

        temp=0;
        imf >> temp;
        e_sctheta.resize(temp);
        for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_sctheta.at(i);
    }
    imf.close();

    // Calibration parameters
    std::ifstream iuf(folder + modelname + ".udraws");
    std::vector<double> e_udraws((nd+1)*pu);
    for(size_t i=0;i<((nd+1)*pu);i++) iuf >> std::scientific >> e_udraws.at(i);
    iuf.close();
    std::vector<double> u0, ucur;
    
    // Initialize uvec and the xf's
    for(size_t j=0;j<pu;j++){u0.push_back(e_udraws[j]);cout << "u0 = " << u0[0] << endl;}
    uvec.setucur(u0);
    uvec.updatex(xp,ucols,p,nf);
    uvec.updatex(xf,ucols,p,nf);

    //Eigen objects for computer model and discrepancy
    mxd fiter(2,np); //Eigen matrix to store the eta and delta at each iteration -- will be reset to zero prior to running get thetavec method  
    mxd fdraw(nd,np); //Eigen matrix to hold posterior draws for eta and delta -- used when writing to the file for ease of notation
    std::vector<mxd, Eigen::aligned_allocator<mxd>> flist(2); //An std vector of dim 2 -- each element is an nd X np eigen matrix

    // Initialize both matrices
    for(size_t i=0; i<2; i++){
        flist[i] = mxd::Zero(nd,np);
    }

    //objects where we'll store the realizations
    std::vector<std::vector<double> > tedraw(nd,std::vector<double>(np));
    std::vector<std::vector<double> > tedrawhf(nd,std::vector<double>(np));
    std::vector<std::vector<double> > tedrawhc(nd,std::vector<double>(np));

    double *fp = new double[np];    
    double *fpf = new double[nf];
    double *fpc = new double[nc];
    dinfo dip,dipf,dipc;
    dip.x = &xp[0]; dip.y=fp; dip.p = p; dip.n=np; dip.tc=1;
    dipf.x = &xf[0]; dipf.y=fpf; dipf.p = p; dipf.n=nf; dipf.tc=1;
    dipc.x = &xc[0]; dipc.y=fpc; dipc.p = p; dipc.n=nc; dipc.tc=1;

    // Temporary vectors used for loading one model realization at a time.
    // Mean trees
    std::vector<int> onn(m,1);
    std::vector<std::vector<int> > oid(m, std::vector<int>(1));
    std::vector<std::vector<int> > ov(m, std::vector<int>(1));
    std::vector<std::vector<int> > oc(m, std::vector<int>(1));
    std::vector<std::vector<double> > otheta(m, std::vector<double>(1));
    // Product variance trees for field data
    std::vector<int> sfnn(mh,1);
    std::vector<std::vector<int> > sfid(mh, std::vector<int>(1));
    std::vector<std::vector<int> > sfv(mh, std::vector<int>(1));
    std::vector<std::vector<int> > sfc(mh, std::vector<int>(1));
    std::vector<std::vector<double> > sftheta(mh, std::vector<double>(1));
    // Product variance trees for model runs
    std::vector<int> scnn(mh,1);
    std::vector<std::vector<int> > scid(mh, std::vector<int>(1));
    std::vector<std::vector<int> > scv(mh, std::vector<int>(1));
    std::vector<std::vector<int> > scc(mh, std::vector<int>(1));
    std::vector<std::vector<double> > sctheta(mh, std::vector<double>(1));
    
    // Draw realizations of the posterior predictive.
    size_t curdx=0;
    size_t cumdx=0;
    size_t k = 2;

#ifdef _OPENMPI
    double tstart=0.0,tend=0.0;
    if(mpirank==0) tstart=MPI_Wtime();
#endif
    // Mean trees first
    if(mpirank==0) cout << "Drawing mean response from posterior predictive" << endl;
    for(size_t i=0;i<nd;i++) {
        curdx=0;
        for(size_t j=0;j<m;j++) {
            onn[j]=e_ots.at(i*m+j);
            oid[j].resize(onn[j]);
            ov[j].resize(onn[j]);
            oc[j].resize(onn[j]);
            otheta[j].resize(onn[j]*k);
            for(size_t l=0;l<(size_t)onn[j];l++) {
                oid[j][l]=e_oid.at(cumdx+curdx+l);
                ov[j][l]=e_ovar.at(cumdx+curdx+l);
                oc[j][l]=e_oc.at(cumdx+curdx+l);
                for(size_t r=0;r<k;r++){
                    otheta[j][l*k+r]=e_otheta.at((cumdx+curdx+l)*k+r);
                    //cout << "Theta_i = " << otheta[j][l*k+r] << endl;
                }   
            }
            curdx+=(size_t)onn[j];
        }
        cumdx+=curdx;

        acb.loadtree_vec(0,m,onn,oid,ov,oc,otheta); 
        //acb.predict_vec(&dip, &fi_test);
        //for(size_t j=0;j<np;j++) tedraw[i][j] = fp[j] + yfmean;

        // Predict eta and delta separately
        acb.predict_thetavec(&dip,&fiter);

        //Store these weights into the Vector of Eigen Matrices
        for(size_t j = 0;j<2; j++){
            flist[j].row(i) = fiter.row(j); //populate the ith row of each flist[j] matrix (ith post draw) for item j (eta or delta)
        }

        // Load the next u
        ucur.clear();
        for(size_t j=0;j<pu;j++){ ucur.push_back(e_udraws[pu*(i+1)+j]);}
        uvec.setucur(ucur);
        uvec.updatex(xp,ucols,p,nf);        
    }

    // Field data Variance trees second
    if(mpirank==0) cout << "Drawing field data sd response from posterior predictive" << endl;
    cumdx=0;
    curdx=0;
    for(size_t i=0;i<nd;i++) {
        curdx=0;
        for(size_t j=0;j<mh;j++) {
            sfnn[j]=e_sfts.at(i*mh+j);
            sfid[j].resize(sfnn[j]);
            sfv[j].resize(sfnn[j]);
            sfc[j].resize(sfnn[j]);
            sftheta[j].resize(sfnn[j]);
            for(size_t k=0;k< (size_t)sfnn[j];k++) {
                sfid[j][k]=e_sfid.at(cumdx+curdx+k);
                sfv[j][k]=e_sfvar.at(cumdx+curdx+k);
                sfc[j][k]=e_sfc.at(cumdx+curdx+k);
                sftheta[j][k]=e_sftheta.at(cumdx+curdx+k);
            }
            curdx+=(size_t)sfnn[j];
        }
        cumdx+=curdx;

        psbmf.loadtree(0,mh,sfnn,sfid,sfv,sfc,sftheta);
        // draw realization
        psbmf.predict(&dipf);
        for(size_t j=0;j<nf;j++) tedrawhf[i][j] = fpf[j];

        // Load the next u
        ucur.clear();
        for(size_t j=0;j<pu;j++){ ucur.push_back(e_udraws[pu*(i+1)+j]);}
        uvec.setucur(ucur);
        uvec.updatex(xf,ucols,p,nf);
   }

    if(nc>0){
        // Model runs Variance trees second
        if(mpirank==0) cout << "Drawing model runs sd response from posterior predictive" << endl;
        cumdx=0;
        curdx=0;
        for(size_t i=0;i<nd;i++) {
            curdx=0;
            for(size_t j=0;j<mh;j++) {
                scnn[j]=e_sfts.at(i*mh+j);
                scid[j].resize(scnn[j]);
                scv[j].resize(scnn[j]);
                scc[j].resize(scnn[j]);
                sctheta[j].resize(scnn[j]);
                for(size_t k=0;k< (size_t)scnn[j];k++) {
                    scid[j][k]=e_scid.at(cumdx+curdx+k);
                    scv[j][k]=e_scvar.at(cumdx+curdx+k);
                    scc[j][k]=e_scc.at(cumdx+curdx+k);
                    sctheta[j][k]=e_sctheta.at(cumdx+curdx+k);
                }
                curdx+=(size_t)sfnn[j];
            }
            cumdx+=curdx;

            psbmc.loadtree(0,mh,scnn,scid,scv,scc,sctheta);
            // draw realization
            psbmc.predict(&dipc);
            for(size_t j=0;j<nc;j++) tedrawhc[i][j] = fpc[j];
        }
    }
   #ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Posterior predictive draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif
    // Save the draws.
    if(mpirank==0) cout << "Saving posterior predictive draws of eta...";
    std::ofstream omf(folder + modelname + ".etadraws" + std::to_string(mpirank));
    fdraw = flist[0];
    for(size_t i=0;i<nd;i++) {
        for(size_t j=0;j<np;j++)
            omf << std::scientific << fdraw(i,j) << " ";
        omf << endl;
    }
    omf.close();

    if(mpirank==0) cout << "Saving posterior predictive draws of delta...";
    std::ofstream odf(folder + modelname + ".deltadraws" + std::to_string(mpirank));
    fdraw = flist[1];
    for(size_t i=0;i<nd;i++) {
        for(size_t j=0;j<np;j++)
            odf << std::scientific << fdraw(i,j) << " ";
        odf << endl;
    }
    odf.close();

    std::ofstream smf(folder + modelname + ".sfdraws" + std::to_string(mpirank));
    for(size_t i=0;i<nd;i++) {
        for(size_t j=0;j<nf;j++)
            smf << std::scientific << tedrawhf[i][j] << " ";
        smf << endl;
    }
    smf.close();

    if(nc>0){
        std::ofstream smfc(folder + modelname + ".scdraws" + std::to_string(mpirank));
        for(size_t i=0;i<nd;i++) {
            for(size_t j=0;j<nc;j++)
                smfc << std::scientific << tedrawhc[i][j] << " ";
            smfc << endl;
        }
        smfc.close();
    }

   if(mpirank==0) cout << " done." << endl;

   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   MPI_Finalize();
#endif
   return 0;
}