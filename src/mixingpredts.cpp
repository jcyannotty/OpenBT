/*
Name: Command Line Interface for BART Bayesian Model Mixing Predictions - Two Step Process
Auth: JCY (yannotty.1@buckeyemail.osu.edu)
Desc: Takes trained model from mixingts.cpp and returns predictions
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

#define MODEL_BARTBMMM 10 //Skipped 8 because MERCK is 8 in cli.cpp


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

    //number of predictors and number of models
    size_t p, k;
    conf >> p;
    conf >> k;

    // Type of prediction to run
    std::string domdrawsstr,dosdrawsstr,dopdrawsstr; 
    bool domdraws, dosdraws, dopdraws; 
    conf >> domdrawsstr;
    conf >> dosdrawsstr;
    conf >> dopdrawsstr;
    if(domdrawsstr == "True" || domdrawsstr == "TRUE"){domdraws = true;}else{domdraws = false;}
    if(dosdrawsstr == "True" || dosdrawsstr == "TRUE"){dosdraws = true;}else{dosdraws = false;}
    if(dopdrawsstr == "True" || dopdrawsstr == "TRUE"){dopdraws = true;}else{dopdraws = false;}

    // Batch size
    size_t batchsize;
    size_t numbatches;
    conf >> batchsize;
    conf >> numbatches;

    //thread count
    int tc;
    conf >> tc;

    // random path
    std::string rpath_str, rpg;
    bool rpath = false;
    conf >> rpath_str;
    conf >> rpg; // gamma root
    if(rpath_str == "True" || rpath_str == "TRUE"){rpath = true;} 

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
        cout << "OpenBT Model Mixing Prediction Interface" << endl;
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
    std::ifstream xf(xfs);
    while(xf >> xtemp)
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
    //Initialize f matrix and make finfo -- used only for model mixing 
    std::vector<double> fpd;
    double ftemp;
    finfo fi_test(np,k);
    std::stringstream ffss;
    std::string ffs;
    ffss << folder << fpcore << mpirank; //get the file to read in -- every processor reads in a different file (dictated by mpirank and R files)
    ffs=ffss.str();
    std::ifstream ff(ffs);
    while(ff >> ftemp)
        fpd.push_back(ftemp);
    Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> fi_test_temp(fpd.data(),k,np);
    fi_test = fi_test_temp.transpose();

#ifndef SILENT
    cout << "&&& made finfo for test data\n";
#endif

    //--------------------------------------------------
    // set up amxbrt object
    amxbrt axb(m);
    axb.setxi(&xi); //set the cutpoints for this model object
    axb.setfi(&fi_test, k); //set the function output for this model object -- main use is to set k 
    //if(fdiscrepancy) {axb.setfdelta(&fdeltamean, &fdeltasd);}  //set individual function discrepacnies if provided -- main use is to set fdiscrepancy to TRUE
   
    //setup psbrt object
    psbrt psbm(mh);
    psbm.setxi(&xi); //set the cutpoints for this model object
   
    //load from file
#ifndef SILENT
    if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif

    // Temp vectors for storage
    std::vector<int> onn(m,1);
    std::vector<std::vector<int> > oid(m, std::vector<int>(1));
    std::vector<std::vector<int> > ov(m, std::vector<int>(1));
    std::vector<std::vector<int> > oc(m, std::vector<int>(1));
    std::vector<std::vector<double> > otheta(m, std::vector<double>(1));
    std::vector<std::vector<double> > optheta(m, std::vector<double>(1)); // projected theta

    //objects where we'll store the realizations
    std::vector<std::vector<double> > tedraw(nd,std::vector<double>(np));
    std::vector<std::vector<double> > tedrawh(nd,std::vector<double>(np));
    std::vector<std::vector<double> > tedrawp(nd,std::vector<double>(np));
    double *fp = new double[np];
    dinfo dip;
    dip.x = &xp[0]; dip.y=fp; dip.p = p; dip.n=np; dip.tc=1;

    //----------------------------------------------
    // Random Path, read in gamma
    //----------------------------------------------
    std::ifstream igf(folder + modelname + ".rpg");
    std::vector<double> e_gamma(nd*m);
    if(rpath){
        for(size_t i=0;i<e_gamma.size();i++) igf >> std::scientific >> e_gamma.at(i);
    }

    //----------------------------------------------
    // Draw realizations of the posterior predictive (batch mode)
    //----------------------------------------------
    size_t ind,im,imh;
    size_t temp = 0;
    size_t curdx=0;
    size_t cumdx=0;
    std::vector<double> ctheta, ptheta;

    //Create Eigen objects to store the weight posterior draws -- these are just the thetas for each bottom node
    mxd wts_iter(k,np); //Eigen matrix to store the weights at each iteration -- will be reset to zero prior to running get wts method  
    mxd wts_draw(nd,np); //Eigen matrix to hold posterior draws for each model weight -- used when writing to the file for ease of notation
    std::vector<mxd, Eigen::aligned_allocator<mxd>> wts_list(k); //An std vector of dim k -- each element is an nd X np eigen matrix
    
    // For projected weights
    std::vector<mxd, Eigen::aligned_allocator<mxd>> pwts_list(k); 
    mxd pwts_draw(nd,np);


    //Initialize wts_list -- the vector of eigen matrices which will hold the nd X np weight draws
    // Moved to below
    /*
    for(size_t i=0; i<k; i++){
        if(domdraws) wts_list[i] = mxd::Zero(nd,np);
        if(dopdraws) pwts_list[i] = mxd::Zero(nd,np);
    }
    */

    #ifdef _OPENMPI
    double tstart=0.0,tend=0.0;
    if(mpirank==0) tstart=MPI_Wtime();
    #endif
    
    if(domdraws || dopdraws){
        std::ifstream imf(folder + modelname + ".fit");
    if(mpirank==0) cout << "Drawing mean response from posterior predictive" << endl;
        for(size_t b=0;b<numbatches;b++){
            imf >> ind;
            imf >> im;
#ifdef _OPENMPI
            if(batchsize!=ind && (nd%batchsize)!=ind) { cout << "Error loading posterior trees" << "nd = " << nd << " -- ind = " << ind << endl; MPI_Finalize(); return 0; }
            if(m!=im) { cout << "Error loading posterior trees" << "m = " << m << " -- im = " << im<< endl; MPI_Finalize(); return 0; }
            //if(mh!=imh) { cout << "Error loading posterior trees"  << endl; MPI_Finalize(); return 0; }
#else
            if(batchsize!=ind && (nd%batchsize)!=ind) { cout << "Error loading posterior trees" << "nd = " << nd << " -- ind = " << ind << endl; return 0; }
            if(m!=im) { cout << "Error loading posterior trees" << "m = " << m << " -- im = " << im<< endl; return 0; }
            //if(mh!=imh) { cout << "Error loading posterior trees"  << endl; return 0; }
#endif
            temp=0;
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

            // Reset and resize things
            wts_draw.resize(ind,np);
            pwts_draw.resize(ind,np);
            for(size_t l=0; l<k; l++){
                if(domdraws) wts_list[l] = mxd::Zero(ind,np);
                if(dopdraws) pwts_list[l] = mxd::Zero(ind,np);
            }
            cumdx=0; // reset cumdx since this is batch looping
            
            for(size_t i=0;i<ind;i++) {
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
                        if(dopdraws){
                            // Project the theta's onto the simplex
                            ctheta.clear();
                            ptheta.clear();
                            optheta[j].resize(onn[j]*k);
                            for(size_t r=0;r<k;r++){
                                otheta[j][l*k+r]=e_otheta.at((cumdx+curdx+l)*k+r);
                                ctheta.push_back(e_otheta.at((cumdx+curdx+l)*k+r));
                            }
                            
                            axb.project_thetavec(&ctheta,ptheta);
                            //optheta[j].insert(optheta[j].end(),ptheta.begin(),ptheta.end());
                            for(size_t r=0;r<k;r++){optheta[j][l*k+r] = ptheta[r];}
                                                        
                        }else{
                            // No simplex projection
                            for(size_t r=0;r<k;r++){
                                otheta[j][l*k+r]=e_otheta.at((cumdx+curdx+l)*k+r);
                            }
                        }
                    }
                    curdx+=(size_t)onn[j];
                }
                cumdx+=curdx;

                if(domdraws){
                    // Load tree and get results for unconstrained data
                    axb.loadtree_vec(0,m,onn,oid,ov,oc,otheta); 
                
                    //Get the current posterior draw of the weights
                    wts_iter = mxd::Zero(k,np);
                    if(!rpath){
                        axb.predict_thetavec(&dip, &wts_iter);
                    }else{
                        // Get and set gamma, then get predictions and clear the temp vec
                        std::vector<double> tempgam;
                        for(size_t j=0;j<m;j++){tempgam.push_back(e_gamma[m*(i+b*batchsize)+j]);}        
                        axb.setgamma(tempgam);
                        axb.predict_thetavec_rpath(&dip, &wts_iter);
                        tempgam.clear();
                    }

                    // Get the resulting prediction
                    for(size_t j=0;j<np;j++){
                        tedraw[i+b*batchsize][j] = fi_test.row(j)*wts_iter.col(j);
                    } 

                    //Store these weights into the Vector of Eigen Matrices
                    for(size_t j = 0; j<k; j++){
                        //wts_list[j].row(i+b*batchsize) = wts_iter.row(j); //populate the ith row of each wts_list[j] matrix (ith post draw) for model weight j
                        wts_list[j].row(i) = wts_iter.row(j);
                    }
                }

                
                if(dopdraws){
                    // Load tree and get results for unconstrained data
                    axb.loadtree_vec(0,m,onn,oid,ov,oc,optheta); 
                    //Get the current posterior draw of the weights
                    wts_iter = mxd::Zero(k,np);
                    if(!rpath){
                        axb.predict_thetavec(&dip, &wts_iter);
                    }else{
                        // Get and set gamma, then get predictions and clear the temp vec
                        std::vector<double> tempgam;
                        for(size_t j=0;j<m;j++){tempgam.push_back(e_gamma[m*(i+b*batchsize)+j]);}        
                        axb.setgamma(tempgam);
                        axb.predict_thetavec_rpath(&dip, &wts_iter);
                        tempgam.clear();
                    }

                    // Get the resulting prediction
                    for(size_t j=0;j<np;j++){
                        tedrawp[i+b*batchsize][j] = fi_test.row(j)*wts_iter.col(j);
                    } 
                    //Store these weights into the Vector of Eigen Matrices
                    for(size_t j = 0; j<k; j++){
                        //pwts_list[j].row(i+b*batchsize) = wts_iter.row(j); //populate the ith row of each wts_list[j] matrix (ith post draw) for model weight j
                        pwts_list[j].row(i) = wts_iter.row(j);
                    }
                }
            }

            if(domdraws){
                if(mpirank==0) cout << "Saving posterior predictive draws...";
                std::ofstream omf;
                std::string rankstr;
                if(mpirank<10){rankstr = "0"+std::to_string(mpirank);}else{rankstr = std::to_string(mpirank);}
                if(b == 0){
                    // Create the file 
                    omf.open(folder + modelname + ".mdraws" + rankstr);
                }else{
                    // Append to the existing file
                    omf.open(folder + modelname + ".mdraws" + rankstr, std::ios_base::app);
                }
                //std::ofstream omf(folder + modelname + ".mdraws" + std::to_string(mpirank));
                
                for(size_t i=0;i<ind;i++) {
                    for(size_t j=0;j<np;j++)
                        omf << std::scientific << tedraw[i][j] << " ";
                    omf << endl;
                }
                omf.close();

                if(mpirank==0) cout << "Saving posterior weight draws..." << endl;
                for(size_t l = 0; l<k; l++){
                    //std::ofstream owf(folder + modelname + ".w" + std::to_string(l+1) + "draws" + std::to_string(mpirank));
                    std::ofstream owf;
                    if(b == 0){
                        // Create the file 
                        owf.open(folder + modelname + ".w" + std::to_string(l+1) + "draws" + rankstr);
                    }else{
                        // Append to the existing file
                        owf.open(folder + modelname + ".w" + std::to_string(l+1) + "draws" + rankstr, std::ios_base::app);
                    }
    
                    wts_draw = wts_list[l];
                    for(size_t i=0;i<ind;i++) {
                        for(size_t j=0;j<np;j++)
                            owf << std::scientific << wts_draw(i,j) << " ";
                        owf << endl;
                    }
                    owf.close();
                }
            }

            // Write the projected weights and ensuing predictions
            if(dopdraws){
                std::string rankstr;
                if(mpirank<10){rankstr = "0"+std::to_string(mpirank);}else{rankstr = std::to_string(mpirank);}
                if(mpirank==0) cout << "Saving projections of posterior predictive draws...";
                //std::ofstream opmf(folder + modelname + ".pmdraws" + std::to_string(mpirank));
                std::ofstream opmf;
                if(b == 0){
                    // Create the file 
                    opmf.open(folder + modelname + ".pmdraws" + rankstr);
                }else{
                    // Append to the existing file
                    opmf.open(folder + modelname + ".pmdraws" + rankstr, std::ios_base::app);
                }
                
                for(size_t i=0;i<ind;i++) {
                    for(size_t j=0;j<np;j++)
                        opmf << std::scientific << tedrawp[i][j] << " ";
                    opmf << endl;
                }
                opmf.close();

                if(mpirank==0) cout << "Saving projections of posterior weight draws..." << endl;
                for(size_t l = 0; l<k; l++){
                    //std::ofstream opwf(folder + modelname + ".pw" + std::to_string(l+1) + "draws" + std::to_string(mpirank));
                    std::ofstream opwf;
                    if(b == 0){
                        // Create the file 
                        opwf.open(folder + modelname + ".pw" + std::to_string(l+1) + "draws" + rankstr);
                    }else{
                        // Append to the existing file
                        opwf.open(folder + modelname + ".pw" + std::to_string(l+1) + "draws" + rankstr, std::ios_base::app);
                    }
                    pwts_draw = pwts_list[l];
                    for(size_t i=0;i<ind;i++) {
                        for(size_t j=0;j<np;j++)
                            opwf << std::scientific << pwts_draw(i,j) << " ";
                        opwf << endl;
                    }
                    opwf.close();
                }
            }
        }
        imf.close();
    }


    //----------------------------------------------
    // Variance trees second
    //----------------------------------------------
    // Temporary vectors used for loading one model realization at a time.
    std::vector<int> snn(mh,1);
    std::vector<std::vector<int> > sid(mh, std::vector<int>(1));
    std::vector<std::vector<int> > sv(mh, std::vector<int>(1));
    std::vector<std::vector<int> > sc(mh, std::vector<int>(1));
    std::vector<std::vector<double> > stheta(mh, std::vector<double>(1));

    if(dosdraws){
        std::ifstream isf(folder + modelname + ".sfit");
        isf >> nd;
        isf >> imh;
        isf >> temp;
        std::vector<int> e_sts(temp);
        for(size_t i=0;i<temp;i++) isf >> e_sts.at(i);

        temp=0;
        isf >> temp;
        std::vector<int> e_sid(temp);
        for(size_t i=0;i<temp;i++) isf >> e_sid.at(i);

        temp=0;
        isf >> temp;
        std::vector<int> e_svar(temp);
        for(size_t i=0;i<temp;i++) isf >> e_svar.at(i);

        temp=0;
        isf >> temp;
        std::vector<int> e_sc(temp);
        for(size_t i=0;i<temp;i++) isf >> e_sc.at(i);

        temp=0;
        isf >> temp;
        std::vector<double> e_stheta(temp);
        for(size_t i=0;i<temp;i++) isf >> std::scientific >> e_stheta.at(i);
        isf.close();

        // Get the sdraws
        if(mpirank==0) cout << "Drawing sd response from posterior predictive" << endl;
        cumdx=0;
        curdx=0;
        for(size_t i=0;i<nd;i++) {
            curdx=0;
            for(size_t j=0;j<mh;j++) {
                snn[j]=e_sts.at(i*mh+j);
                sid[j].resize(snn[j]);
                sv[j].resize(snn[j]);
                sc[j].resize(snn[j]);
                stheta[j].resize(snn[j]);
                for(size_t k=0;k< (size_t)snn[j];k++) {
                    sid[j][k]=e_sid.at(cumdx+curdx+k);
                    sv[j][k]=e_svar.at(cumdx+curdx+k);
                    sc[j][k]=e_sc.at(cumdx+curdx+k);
                    stheta[j][k]=e_stheta.at(cumdx+curdx+k);
                }
                curdx+=(size_t)snn[j];
            }
            cumdx+=curdx;

            psbm.loadtree(0,mh,snn,sid,sv,sc,stheta);
            // draw realization
            psbm.predict(&dip);
            for(size_t j=0;j<np;j++) tedrawh[i][j] = fp[j];
        }
    }

    #ifdef _OPENMPI
    if(mpirank==0) {
        tend=MPI_Wtime();
        cout << "Posterior predictive draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
    }
    #endif

    //----------------------------------------------
    // Save the draws
    //----------------------------------------------    
/*
    if(domdraws){
        if(mpirank==0) cout << "Saving posterior predictive draws...";
        std::ofstream omf(folder + modelname + ".mdraws" + std::to_string(mpirank));
        for(size_t i=0;i<nd;i++) {
            for(size_t j=0;j<np;j++)
                omf << std::scientific << tedraw[i][j] << " ";
            omf << endl;
        }
        omf.close();

        if(mpirank==0) cout << "Saving posterior weight draws..." << endl;
        for(size_t l = 0; l<k; l++){
            std::ofstream owf(folder + modelname + ".w" + std::to_string(l+1) + "draws" + std::to_string(mpirank));
            wts_draw = wts_list[l];
            for(size_t i=0;i<nd;i++) {
                for(size_t j=0;j<np;j++)
                    owf << std::scientific << wts_draw(i,j) << " ";
                owf << endl;
            }
            owf.close();
        }
    }

    // Write the projected weights and ensuing predictions
    if(dopdraws){
        if(mpirank==0) cout << "Saving projections of posterior predictive draws...";
        std::ofstream opmf(folder + modelname + ".pmdraws" + std::to_string(mpirank));
        for(size_t i=0;i<nd;i++) {
            for(size_t j=0;j<np;j++)
                opmf << std::scientific << tedrawp[i][j] << " ";
            opmf << endl;
        }
        opmf.close();

        if(mpirank==0) cout << "Saving projections of posterior weight draws..." << endl;
        for(size_t l = 0; l<k; l++){
            std::ofstream opwf(folder + modelname + ".pw" + std::to_string(l+1) + "draws" + std::to_string(mpirank));
            pwts_draw = pwts_list[l];
            for(size_t i=0;i<nd;i++) {
                for(size_t j=0;j<np;j++)
                    opwf << std::scientific << pwts_draw(i,j) << " ";
                opwf << endl;
            }
            opwf.close();
        }
    }
*/
    if(dosdraws){
        if(mpirank==0) cout << "Saving posterior standard dev. draws..." << endl;
        std::ofstream smf(folder + modelname + ".sdraws" + std::to_string(mpirank));
        for(size_t i=0;i<nd;i++) {
            for(size_t j=0;j<np;j++)
                smf << std::scientific << tedrawh[i][j] << " ";
            smf << endl;
        }
        smf.close();
    }

    if(mpirank==0) cout << " done." << endl;

    //-------------------------------------------------- 
    // Cleanup.
#ifdef _OPENMPI
    MPI_Finalize();
#endif
    return 0;
}

