#include <chrono>
#include <iostream>
#include <string>
#include <ctime>
#include <sstream>
#include <map>

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
#include "parameters.h"
//#include "calibratefuns.cpp"

using std::cout;
using std::endl;

#define MODEL_BT 1
#define MODEL_BINOMIAL 2
#define MODEL_POISSON 3
#define MODEL_BART 4
#define MODEL_HBART 5
#define MODEL_PROBIT 6
#define MODEL_MODIFIEDPROBIT 7
#define MODEL_MERCK_TRUNCATED 8
#define MODEL_MIXBART 9
#define MODEL_MIXEMULATE 10
#define MODEL_MIXCALIBRATE 11


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

    //Number of models
    int nummodels;
    conf >> nummodels;

    // extract the cores for the mix model and simulator inputs
    std::string xcore,ycore,score,ccore;
    double means,base,baseh,power,powerh,lam,nu;
    size_t m,mh,minnumbot;

    std::vector<std::string> xcore_list,ycore_list,score_list,ccore_list;
    std::vector<double> means_list,base_list,baseh_list,power_list,powerh_list, lam_list, nu_list;
    std::vector<size_t> m_list, mh_list, minnumbot_list;
    
    // Get model mixing cores & inputs then get each emulator cores & inputs 
    for(int i=0;i<=nummodels;i++){
        // Get xcore, ycore, score, and ccore
        conf >> xcore;
        conf >> ycore;
        conf >> score;
        conf >> ccore;

        // Store into respective lists
        xcore_list.push_back(xcore);
        ycore_list.push_back(ycore);
        score_list.push_back(score);
        ccore_list.push_back(ccore);

        // Data means
        conf >> means;
        means_list.push_back(means);

        // Tree sizes
        conf >> m;
        conf >> mh;
        m_list.push_back(m);
        mh_list.push_back(mh);

        // Tree prior hyperparameters
        conf >> base;
        conf >> baseh;
        conf >> power;
        conf >> powerh;
        base_list.push_back(base);
        baseh_list.push_back(baseh);
        power_list.push_back(power);
        powerh_list.push_back(powerh);

        // Variance Prior
        conf >> lam;
        conf >> nu;
        lam_list.push_back(lam);
        nu_list.push_back(nu);

        // minnode size
        conf >> minnumbot;
        minnumbot_list.push_back(minnumbot);

        // Prints
        /*
        cout << "xcore = " << xcore << endl;
        cout << "ycore = " << ycore << endl;
        cout << "score = " << score << endl;
        cout << "ccore = " << ccore << endl;
        cout << "data.mean = " << means << endl;
        cout << "m = " << m << endl;
        cout << "mh = " << mh << endl;
        cout << "base = " << base << endl;
        cout << "baseh = " << baseh << endl;
        cout << "power = " << power << endl;
        cout << "powerh = " << powerh << endl;
        cout << "lam = " << lam << endl;
        cout << "nu = " << nu << endl;
        cout << "minnumbot = " << minnumbot << endl;
        */
    }

    // Get the design columns per emulator
    std::vector<std::vector<size_t>> x_cols_list(nummodels, std::vector<size_t>(1));
    std::vector<std::vector<size_t>> u_cols_list(nummodels, std::vector<size_t>(1));
    std::map<size_t,std::vector<size_t>> xmap, umap;
    std::vector<size_t> pvec, qvec;
    size_t p, xcol, q, ucol, px, pu;
    for(int i=0;i<nummodels;i++){
        conf >> p;
        pvec.push_back(p);
        x_cols_list[i].resize(p);
        for(size_t j = 0; j<p; j++){
            conf >> xcol;
            x_cols_list[i][j] = xcol - 1;
            xmap[xcol].push_back(i);
        }
    }
    px = xmap.size();

    // Get the design columns per emulator for theta
    if(modeltype == MODEL_MIXCALIBRATE){
        for(int i=0;i<nummodels;i++){
            conf >> q;
            qvec.push_back(q);
            u_cols_list[i].resize(q);
            for(size_t j = 0; j<q; j++){
                conf >> ucol;
                u_cols_list[i][j] = ucol - 1; // store the column number in the ith model's vector
                umap[ucol].push_back(i); // store the ith model's index in the calibration map
            }
        }
    }else{
        // Else read in the place holder for tc_cols_list, which is just zero
        conf >> q;
        qvec.resize(nummodels,0);
    }
    pu = umap.size(); // total number of parameters
    //cout << "pu = " << pu << endl; 

    // Get the id root computer and field obs ids per emulator
    // std::string idcore;
    // conf >> idcore;

    // MCMC properties
    size_t nd, nburn, nadapt, adaptevery;
    conf >> nd;
    conf >> nburn;
    conf >> nadapt;
    conf >> adaptevery;

    // Get tau and beta for terminal node priors
    std::vector<double> tau_emu_list;
    double tau_disc, tau_wts, tau_emu, beta_disc, beta_wts;
    conf >> tau_disc; //discrepancy tau
    conf >> tau_wts; // wts tau
    for(int i=0;i<nummodels;i++){
        conf >> tau_emu; // emulator tau
        tau_emu_list.push_back(tau_emu);
    }
    conf >> beta_disc;
    conf >> beta_wts;

    //control
    double pbd, pb, pbdh, pbh;
    double stepwpert, stepwperth;
    double probchv, probchvh;
    int tc;
    //size_t minnumbot;
    //size_t minnumboth;
    size_t printevery;
    std::string xicore, cpcore;
    std::string modelname;
    conf >> pbd;
    conf >> pb;
    conf >> pbdh;
    conf >> pbh;
    conf >> stepwpert;
    conf >> stepwperth;
    conf >> probchv;
    conf >> probchvh;
    //conf >> minnumbot;
    //conf >> minnumboth;
    conf >> printevery;
    conf >> xicore;
    conf >> cpcore;
    conf >> tc;
    conf >> modelname;

    bool dopert=true;
    bool doperth=true;
    if(probchv<0) dopert=false;
    if(probchvh<0) doperth=false;
    
    // summary statistics yes/no
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
        cout << "OpenBT model mixing interface" << endl;
        cout << "Loading config file at " << folder << endl;
    }

    //--------------------------------------------------
    //read in y for mixing and z's for emulation
    std::vector<std::vector<double>> y_list(ycore_list.size(), std::vector<double>(1));
    std::vector<double> y;
    std::vector<size_t> nvec(nummodels+1,0);
    double ytemp;
    size_t n=0;
    std::stringstream yfss;
    std::string yfs;
    std::ifstream yf;

    for(size_t i=0;i<ycore_list.size();i++){
    if(y.size()>0){y.clear();} //clear the contents of the y vector
    #ifdef _OPENMPI
        if(mpirank>0) { //only load data on slaves
    #endif
        yfss << folder << ycore_list[i] << mpirank;
        yfs=yfss.str();
        yf.open(yfs);
        while(yf >> ytemp)
            y.push_back(ytemp);
        n=y.size();
        // Store into the vectors
        //nvec.push_back(n);
        nvec[i]=n;
       
        y_list[i].resize(n);
        y_list[i] = y;
        
        //reset stream variables
        yfss.str("");
        yf.close();
    
    #ifndef SILENT
        cout << "node " << mpirank << " loaded " << n << " from " << yfs <<endl;
    #endif
    #ifdef _OPENMPI
    }
    #endif
    }

    //--------------------------------------------------
    //read in x 
    std::vector<std::vector<double>> x_list(xcore_list.size(), std::vector<double>(1));
    std::vector<std::vector<double>> xp_list; // x inputs for the variance model (stores )
    std::vector<double> x;
    std::stringstream xfss;
    std::string xfs;
    std::ifstream xf;     
    double xtemp;
    p = 0;
    for(size_t i = 0;i<xcore_list.size();i++){
#ifdef _OPENMPI
        if(mpirank>0) {
#endif
        if(x.size() > 0){x.clear();}
        xfss << folder << xcore_list[i] << mpirank;
        xfs=xfss.str();
        xf.open(xfs);
        while(xf >> xtemp){
            x.push_back(xtemp);
            //cout << "model = " << i << "---- x = " << xtemp << endl;
        }
        p = x.size()/nvec[i];
        if(i == 0){
            pvec.insert(pvec.begin(), p); // The p for the mixing inputs is not read in from R, hence find it now
            qvec.insert(qvec.begin(), 0); // place holder, 0 calibration parameters are used in the weight models
        }

        // Update pvec - this works for any model since the default of qvec is 0
        pvec[i] = pvec[i] + qvec[i];

        // Store into the vectors
        x_list[i].resize(nvec[i]*pvec[i]); //qvec[i] = 0 if not using calibration 
        x_list[i] = x;

        //reset stream variables
        xfss.str("");
        xf.close();
#ifndef SILENT
        cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << p << " from " << xfs << endl;
#endif
#ifdef _OPENMPI
        }
        int tempp = (unsigned int) pvec[i];
        MPI_Allreduce(MPI_IN_PLACE,&tempp,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if(mpirank>0 && pvec[i] != ((size_t) tempp)) { cout << "PROBLEM LOADING DATA" << endl; MPI_Finalize(); return 0;}
        pvec[i]=(size_t)tempp;
#endif
    }

    //--------------------------------------------------
    // Create xf_list to hold the subset of field obs inputs used for predictions with each emulator
    // Ex: xf_list[0] uses the field inputs in x_list[0] to construct a n x p1 design matrix used to get predictions with emulator 1
    // Ex: xf_list[1] uses the field inputs in x_list[0] to construct a n x p2 design matrix used to get predictions with emulator 2  
    std::vector<std::vector<double>> xf_list(nummodels);
    std::vector<double> utemp;
    size_t xcolsize = 0, ucolsize = 0;
    xcol = 0;
    // Get the appropriate x columns
    if(mpirank > 0){
        for(int i=0;i<nummodels;i++){
            xcolsize = x_cols_list[i].size(); //x_cols_list is nummodel dimensional -- only for emulators
            // Initialize calibration vector
            if(modeltype==MODEL_MIXCALIBRATE){ 
                ucolsize = u_cols_list[i].size(); 
                utemp.resize(ucolsize,0);
            } 
            for(size_t j=0;j<nvec[0];j++){
                for(size_t k=0;k<xcolsize;k++){
                    xcol = x_cols_list[i][k];
                    xf_list[i].push_back(x_list[0][j*xcolsize + xcol]); //xf_list is nummodel dimensional -- only for emulators
                    //cout << "model = " << i+1 << "--x = " << x_list[0][j*xcolsize + xcol] << endl;
                }
                if(modeltype==MODEL_MIXCALIBRATE){
                    xf_list[i].insert(xf_list[i].end(),utemp.begin(),utemp.end());
                }
            }
        }
    }

    //--------------------------------------------------
    //dinfo
    std::vector<dinfo> dinfo_list(nummodels+1);
    for(int i=0;i<=nummodels;i++){
        dinfo_list[i].n=0;dinfo_list[i].p=pvec[i],dinfo_list[i].x = NULL;dinfo_list[i].y=NULL;dinfo_list[i].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0){ 
#endif 
            dinfo_list[i].n=nvec[i]; dinfo_list[i].x = &x_list[i][0]; dinfo_list[i].y = &y_list[i][0];
#ifdef _OPENMPI
        }
#endif
    }

    //--------------------------------------------------
    //read in sigmav  -- same as above.
    std::vector<std::vector<double>> sigmav_list(score_list.size(), std::vector<double>(1));
    std::vector<double> sigmav;
    std::vector<size_t> nsigvec;
    std::vector<dinfo> disig_list(nummodels+1);
    std::vector<double*> sig_vec(nummodels+1);
    std::stringstream sfss;
    std::string sfs;
    std::ifstream sf;
    double stemp;
    size_t nsig=0;
    for(int i=0;i<=nummodels;i++){
#ifdef _OPENMPI
        if(mpirank>0) { //only load data on slaves
#endif
        sigmav.clear(); // clear the vector of any contents
        sfss << folder << score_list[i] << mpirank;
        sfs=sfss.str();
        sf.open(sfs);
        while(sf >> stemp)
            sigmav.push_back(stemp);
        nsig=sigmav.size(); 
        // Store the results in the vector
        sigmav_list[i].resize(nsig);
        sigmav_list[i] = sigmav;
        //reset stream variables
        sfss.str("");
        sf.close();
#ifndef SILENT
        cout << "node " << mpirank << " loaded " << nsig << " from " << sfs <<endl;
#endif
#ifdef _OPENMPI
        if(nvec[i]!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; MPI_Finalize(); return 0; }
        }
#else
        if(nvec[i]!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; return 0; }
#endif

        sig_vec[i]=&sigmav_list[i][0];
        disig_list[i].n=0; disig_list[i].p=pvec[i]; disig_list[i].x=NULL; disig_list[i].y=NULL; disig_list[i].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0) { 
#endif
            disig_list[i].n=nvec[i];disig_list[i].x=&x_list[i][0];disig_list[i].y=sig_vec[i];
    
#ifdef _OPENMPI
        }
#endif
    }

    //--------------------------------------------------
    // read in the initial change of variable rank correlation matrix
    std::vector<std::vector<std::vector<double>>> chgv_list;
    std::vector<std::vector<double>> chgv;
    std::vector<double> cvvtemp;
    double cvtemp;
    std::stringstream chgvfss;
    std::string chgvfs;
    std::ifstream chgvf;
    std::vector<int*> lwr_vec(nummodels+1, new int[tc]);
    std::vector<int*> upr_vec(nummodels+1, new int[tc]);

    for(int k=0;k<=nummodels;k++){
        chgvfss << folder << ccore_list[k];
        chgvfs=chgvfss.str();
        chgvf.open(chgvfs);
        for(size_t i=0;i<dinfo_list[k].p;i++) {
            cvvtemp.clear();
            for(size_t j=0;j<dinfo_list[k].p;j++) {
                chgvf >> cvtemp;
                cvvtemp.push_back(cvtemp);
            }
            chgv.push_back(cvvtemp);
        }
        chgv_list.push_back(chgv);
        //reset stream variables
        chgvfss.str("");
        chgvf.close();
#ifndef SILENT
    cout << "mpirank=" << mpirank << ": change of variable rank correlation matrix loaded:" << endl;
#endif
    }
    // if(mpirank==0) //print it out:
    //    for(size_t i=0;i<di.p;i++) {
    //       for(size_t j=0;j<di.p;j++)
    //          cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
    //       cout << endl;
    //    }

    //--------------------------------------------------
    // decide what variables each slave node will update in change-of-variable proposals.
#ifdef _OPENMPI
    //int* lwr=new int[tc];
    //int* upr=new int[tc];
    for(int j=0;j<=nummodels;j++){
        lwr_vec[j][0]=-1; upr_vec[j][0]=-1;
        for(size_t i=1;i<(size_t)tc;i++) { 
            lwr_vec[j][i]=-1; upr_vec[j][i]=-1; 
            calcbegend(pvec[j],i-1,tc-1,&lwr_vec[j][i],&upr_vec[j][i]);
            if(pvec[j]>1 && lwr_vec[j][i]==0 && upr_vec[j][i]==0) { lwr_vec[j][i]=-1; upr_vec[j][i]=-1; }
        }
#ifndef SILENT
        if(mpirank>0) cout << "Slave node " << mpirank << " will update variables " << lwr_vec[j][mpirank] << " to " << upr_vec[j][mpirank]-1 << endl;
#endif
    }
#endif

    //--------------------------------------------------
    //make xinfo
    std::vector<xinfo> xi_list(nummodels+1);
    std::vector<double> xivec;
    std::stringstream xifss;
    std::string xifs;
    std::ifstream xif;
    double xitemp;
    size_t ind = 0;

    for(int i=0;i<=nummodels;i++){
        xi_list[i].resize(pvec[i]);
        for(size_t j=0;j<pvec[i];j++) {
            // Get the next column in the x_cols_list -- important since emulators may have different inputs
            // ind is used to located the file name
            if(i>0){ 
                // Append x column if j < x_cols_list[i-1].size(), else append calibration columns
                if(j<x_cols_list[i-1].size()){
                    ind = (size_t)x_cols_list[i-1][j] + 1; // from x list (possible values 1,2,...,px)
                }else{
                    ind = (size_t)u_cols_list[i-1][j-x_cols_list[i-1].size()] + 1; // from calibration list, (possible values 1,2,...,pu)
                    ind = ind + px; // ajdustment to get the ordering right (possible values px+1,px+2,...,px+pu)
                }
            }else{
                ind = j+1;
            }
            //cout << "xinfo (i,j) << (" << i << "," << j << "): " << ind << endl;
            xifss << folder << xicore << (ind);
            xifs=xifss.str();
            xif.open(xifs);
            while(xif >> xitemp){
                xivec.push_back(xitemp);
            }
            xi_list[i][j]=xivec;
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
        for(size_t j=0;j<pvec[i];j++){
            cout << "Variable " << i << " has numcuts=" << xi_list[i][j].size() << " : ";
            cout << xi_list[i][j][0] << " ... " << xi_list[i][j][xi_list[i][j].size()-1] << endl;
        }
#endif
    }

    //--------------------------------------------------    
    // Read in priors for thetas if using mixing and calibration
    std::vector<double> avec, bvec, propwidth, u0vec;
    std::vector<std::string> uprior;
    std::stringstream cfss;
    std::string cfs;
    std::ifstream cf;
    double au, bu;
    if(modeltype==MODEL_MIXCALIBRATE){
        cfss << folder << cpcore;
        cfs=cfss.str();
        cf.open(cfs);
        for(size_t i=0;i<pu;i++){
            cf >> au;
            cf >> bu;
            avec.push_back(au);
            bvec.push_back(bu);
            propwidth.push_back((bu-au)*.25); // propsal width
            u0vec.push_back((bu+au)*.5); // initial value -- prior mean
            uprior.push_back("uniform");
        }
        cf.close();
    }

/*
    //--------------------------------------------------    
    // For testing -- read in true wts (testing only)
    std::vector<double> truewts;
    std::stringstream wfss;
    std::string wfs;
    std::ifstream wf;
    double wtemp;
    mxd wts_matrix(nvec[0], nummodels);
    wfss << "/home/johnyannotty/Documents/Model Mixing BART/Spring 2023/Mix_Polynomials/wt" << mpirank;
    wfs=wfss.str();
    wf.open(wfs);
    while(wf >> wtemp){
        truewts.push_back(wtemp);
    }
    
    makefinfo(nummodels,nvec[0],&truewts[0],wts_matrix);
    vxd yhat0(nvec[0]), yhat1(nvec[0]);
*/

    //--------------------------------------------------
    //Set up model objects and MCMC
    //--------------------------------------------------    
    ambrt *ambm_list[nummodels]; //additive mean bart emulators
    psbrt *psbm_list[nummodels]; //product variance for bart emulators
    amxbrt axb(m_list[0]); // additive mean mixing bart
    psbrt pxb(mh_list[0],lam_list[0]); //product model for mixing variance
    std::vector<dinfo> dips_list(nummodels+1); //dinfo for psbrt objects
    std::vector<double*> r_list(nummodels+1); //residual list
    double opm; //variance info
    double lambda; //variance info
    finfo fi;
    size_t tempn; // used when defining dips_list for emulators
    int l = 0; //Used for indexing emulators
    nu = 1.0; //reset nu to 1, previosuly defined earlier in program

    //Initialize the model mixing bart objects
    if(mpirank > 0){
        //fi = mxd::Ones(nvec[0], nummodels+1); //dummy initialize to matrix of 1's -- n0 x K+1 (1st column is discrepancy)
        fi = mxd::Ones(nvec[0], nummodels); //dummy initialize to matrix of 1's -- n0 x K (no discrepancy)
    }
    //cutpoints
    axb.setxi(&xi_list[0]);   
    //function output information
    //axb.setfi(&fi, nummodels+1);
    axb.setfi(&fi, nummodels);
    //data objects
    axb.setdata_vec(&dinfo_list[0]);  //set the data
    //thread count
    axb.settc(tc-1);      //set the number of slaves when using MPI.
    //mpi rank
#ifdef _OPENMPI
    axb.setmpirank(mpirank);  //set the rank when using MPI.
    axb.setmpicvrange(lwr_vec[0],upr_vec[0]); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
    //tree prior
    axb.settp(base_list[0], //the alpha parameter in the tree depth penalty prior
            power_list[0]     //the beta parameter in the tree depth penalty prior
            );
    //MCMC info
    axb.setmi(
            pbd,  //probability of birth/death
            pb,  //probability of birth
            minnumbot_list[0],    //minimum number of observations in a bottom node
            dopert, //do perturb/change variable proposal?
            stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv_list[0]  //initialize the change of variable correlation matrix.
            );
    //Set prior information
    /*
    mxd prior_precision(nummodels+1,nummodels+1);
    vxd prior_mean(nummodels+1);
    prior_precision = (1/(tau_wts*tau_wts))*mxd::Identity(nummodels+1,nummodels+1);
    prior_precision(0,0) = (1/(tau_disc*tau_disc));
    prior_mean = beta_wts*vxd::Ones(nummodels+1);
    prior_mean(0) = beta_disc;
    
    //Sets the model priors for the functions if they are different
    axb.setci(prior_precision, prior_mean, sig_vec[0]);
    */
    mxd prior_precision(nummodels,nummodels);
    vxd prior_mean(nummodels);
    prior_precision = (1/(tau_wts*tau_wts))*mxd::Identity(nummodels,nummodels);
    prior_mean = beta_wts*vxd::Ones(nummodels);
    
    //Sets the model priors for the functions if they are different
    axb.setci(prior_precision, prior_mean, sig_vec[0]);
    
    //--------------------------------------------------
    //setup psbrt object
    //make di for psbrt object
    dips_list[0].n=0; dips_list[0].p=pvec[0]; dips_list[0].x=NULL; dips_list[0].y=NULL; dips_list[0].tc=tc;
    for(int j=0;j<=nummodels;j++) r_list[j] = NULL;
    //double *r = NULL;
#ifdef _OPENMPI
    if(mpirank>0) {
#endif
    r_list[0] = new double[nvec[0]]; 
    //r = new double[nvec[0]];
    for(size_t i=0;i<nvec[0];i++) r_list[0][i]=sigmav_list[0][i];
    //for(size_t i=0;i<nvec[0];i++) r[i]=sigmav_list[0][i];
    dips_list[0].x=&x_list[0][0]; dips_list[0].y=r_list[0]; dips_list[0].n=nvec[0];
    //dips_list[0].x=&x_list[0][0]; dips_list[0].y=r; dips_list[0].n=nvec[0];
#ifdef _OPENMPI
    }
#endif

    //Variance infomration
    opm=1.0/((double)mh_list[0]);
    nu=2.0*pow(nu_list[0],opm)/(pow(nu_list[0],opm)-pow(nu_list[0]-2.0,opm));
    lambda=pow(lam_list[0],opm);
 
    //cutpoints
    pxb.setxi(&xi_list[0]);    //set the cutpoints for this model object
    //data objects
    pxb.setdata(&dips_list[0]);  //set the data
    //thread count
    pxb.settc(tc-1); 
    //mpi rank
#ifdef _OPENMPI
    pxb.setmpirank(mpirank);  //set the rank when using MPI.
    pxb.setmpicvrange(lwr_vec[0],upr_vec[0]); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
    //tree prior
    pxb.settp(baseh_list[0], //the alpha parameter in the tree depth penalty prior
            powerh_list[0]     //the beta parameter in the tree depth penalty prior
            );
    pxb.setmi(
            pbdh,  //probability of birth/death
            pbh,  //probability of birth
            minnumbot_list[0],    //minimum number of observations in a bottom node
            doperth, //do perturb/change variable proposal?
            stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv_list[0]  //initialize the change of variable correlation matrix.
            );
    pxb.setci(nu,lambda);

    //Initialize the emulation bart objects
    for(int j=1;j<=nummodels;j++){
        // Set l for indexing
        l = j-1;
        //Redefine the class instance
        ambm_list[l] = new ambrt(m_list[j]);     
        //cutpoints
        ambm_list[l]->setxi(&xi_list[j]);
        //data objects
        ambm_list[l]->setdata(&dinfo_list[j]);        
        //thread count
        ambm_list[l]->settc(tc-1);      //set the number of slaves when using MPI.
        //mpi rank
    #ifdef _OPENMPI
        ambm_list[l]->setmpirank(mpirank);  //set the rank when using MPI.
        ambm_list[l]->setmpicvrange(lwr_vec[j],upr_vec[j]); //range of variables each slave node will update in MPI change-of-var proposals.
    #endif
        //tree prior
        ambm_list[l]->settp(base_list[j], //the alpha parameter in the tree depth penalty prior
                    power_list[j]     //the beta parameter in the tree depth penalty prior
                    );
        //MCMC info
        ambm_list[l]->setmi(
                pbd,  //probability of birth/death
                pb,  //probability of birth
                minnumbot_list[j],    //minimum number of observations in a bottom node
                dopert, //do perturb/change variable proposal?
                stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
                probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
                &chgv_list[j]  //initialize the change of variable correlation matrix.
                );
        ambm_list[l]->setci(tau_emu_list[l],sig_vec[j]);
        
        //--------------------------------------------------
        //setup psbrt object
        psbm_list[l] = new psbrt(mh_list[j]);

        //make di for psbrt object
        opm=1.0/((double)mh_list[j]);
        nu=2.0*pow(nu_list[j],opm)/(pow(nu_list[j],opm)-pow(nu_list[j]-2.0,opm));
        lambda=pow(lam_list[j],opm);

        //make dips info
        tempn = 0;
        dips_list[j].n=0; dips_list[j].p=p; dips_list[j].x=NULL; dips_list[j].y=NULL; dips_list[j].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0) {
#endif      
            r_list[j] = new double[nvec[j]];
            for(size_t i=0;i<nvec[j];i++) r_list[j][i]=sigmav_list[j][i];
            dips_list[j].x=&x_list[j][0]; dips_list[j].y=r_list[j]; dips_list[j].n=nvec[j];

#ifdef _OPENMPI
        }
#endif
        //cutpoints
        psbm_list[l]->setxi(&xi_list[j]);    //set the cutpoints for this model object
        //data objects
        psbm_list[l]->setdata(&dips_list[j]);  //set the data
        //thread count
        psbm_list[l]->settc(tc-1); 
        //mpi rank
        #ifdef _OPENMPI
        psbm_list[l]->setmpirank(mpirank);  //set the rank when using MPI.
        psbm_list[l]->setmpicvrange(lwr_vec[j],upr_vec[j]); //range of variables each slave node will update in MPI change-of-var proposals.
        #endif
        //tree prior
        psbm_list[l]->settp(baseh_list[j], //the alpha parameter in the tree depth penalty prior
                powerh_list[j]     //the beta parameter in the tree depth penalty prior
                );
        psbm_list[l]->setmi(
                pbdh,  //probability of birth/death
                pbh,  //probability of birth
                minnumbot_list[j],    //minimum number of observations in a bottom node
                doperth, //do perturb/change variable proposal?
                stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
                probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
                &chgv_list[j]  //initialize the change of variable correlation matrix.
                );
        psbm_list[l]->setci(nu,lambda);
    }
    
    //--------------------------------------------------
    // Initialize calibration parameters
    param uvec(pu);
    std::vector<double> udraws; // container for calibration parameters
    if(modeltype==MODEL_MIXCALIBRATE){
#ifdef _OPENMPI
        uvec.setmpirank(mpirank);
        uvec.settc(tc-1);
#endif
        uvec.setpriors(uprior, avec, bvec);
        uvec.setproposals(uprior, propwidth);
        uvec.setucur(u0vec);
        for(int j=0; j<nummodels;j++){
            // Initialize the xf_list calibration parameter values
            uvec.updatexmm(xf_list[j],u_cols_list[j],pvec[j+1],nvec[0]);
        }
        //cout << "ucurr0 = " << uvec.ucur[0] << endl;
        //cout << "ucurr1 = " << uvec.ucur[1] << endl;
    }

    //-------------------------------------------------- 
    // MCMC
    //-------------------------------------------------- 
    // Method Wrappers
    brtMethodWrapper faxb(&brt::f,axb);
    brtMethodWrapper fpxb(&brt::f,pxb);
    //brtMethodWrapper *fambm_list[nummodels];
    //brtMethodWrapper *fpsbm_list[nummodels];
    
    // Define containers -- similar to those in cli.cpp, except now we iterate over K+1 bart objects
    std::vector<std::vector<int>> onn_list(nummodels+1, std::vector<int>(nd,1));
    std::vector<std::vector<std::vector<int>>> oid_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<int>>> ovar_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<int>>> oc_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<double>>> otheta_list(nummodels+1, std::vector<std::vector<double>>(nd, std::vector<double>(1)));
    
    std::vector<std::vector<int>> snn_list(nummodels+1, std::vector<int>(nd,1));
    std::vector<std::vector<std::vector<int>>> sid_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<int>>> svar_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<int>>> sc_list(nummodels+1, std::vector<std::vector<int>>(nd, std::vector<int>(1)));
    std::vector<std::vector<std::vector<double>>> stheta_list(nummodels+1, std::vector<std::vector<double>>(nd, std::vector<double>(1)));
  
    // Initialization of objects
    for(int i=0;i<=nummodels;i++){
        onn_list[i].resize(nd*m_list[i],1);
        oid_list[i].resize(nd*m_list[i], std::vector<int>(1));
        ovar_list[i].resize(nd*m_list[i], std::vector<int>(1));
        oc_list[i].resize(nd*m_list[i], std::vector<int>(1));
        otheta_list[i].resize(nd*m_list[i], std::vector<double>(1));
        

        snn_list[i].resize(nd*mh_list[i],1);
        sid_list[i].resize(nd*mh_list[i], std::vector<int>(1));
        svar_list[i].resize(nd*mh_list[i], std::vector<int>(1));
        sc_list[i].resize(nd*mh_list[i], std::vector<int>(1));
        stheta_list[i].resize(nd*mh_list[i], std::vector<double>(1));
        if(i>0){
            //fambm_list[i-1] = new brtMethodWrapper(&brt::f,*ambm_list[i-1]);
            //fpsbm_list[i-1] = new brtMethodWrapper(&brt::f,*psbm_list[i-1]);
        }
     }

    // dinfo for predictions -- used to get predictions from each emulator at field obs inputs
    // These predictions are fed into the model mixing framework to learn the model weights
    std::vector<dinfo> dimix_list(nummodels);
    std::vector<double*> fmix_list(nummodels);
    for(int i=0;i<nummodels;i++){
        // Initialize class objects
        if(mpirank > 0){
            fmix_list[i] = new double[nvec[0]];
            dimix_list[i].y=fmix_list[i];
            dimix_list[i].p = pvec[i+1]; // pvec[i+1] gives the number of columns for the given emulator 
            dimix_list[i].n=nvec[0]; // Sample size is always nvec[0] = ntrain
            dimix_list[i].tc=1;
            dimix_list[i].x = &xf_list[i][0]; // prediction grid of dimension nvec[0] x pvec[i+1]
        }else{
            fmix_list[i] = NULL;
            dimix_list[i].y = NULL;
            dimix_list[i].x = NULL;
            dimix_list[i].p = pvec[i+1]; 
            dimix_list[i].n=0;
            dimix_list[i].tc=1;
        }
    }

    // dinfo for calibration -- used to get predictions from each emulator at field obs inputs and proposed u
    std::vector<dinfo> dimixprop_list(nummodels);
    std::vector<double*> fmixprop_list(nummodels);
    std::vector<double> propresid;
    finfo fiprop;
    if(modeltype==MODEL_MIXCALIBRATE){
        for(int i=0;i<nummodels;i++){
            // Initialize class objects
            if(mpirank > 0){
                fmixprop_list[i] = new double[nvec[0]];
                dimixprop_list[i].y=fmixprop_list[i]; dimixprop_list[i].p = pvec[i+1];  
                dimixprop_list[i].n=nvec[0];dimixprop_list[i].tc=1;dimixprop_list[i].x = &xf_list[i][0];
                fiprop = mxd::Ones(nvec[0], nummodels);
            }else{
                fmixprop_list[i] = NULL;dimixprop_list[i].y = NULL;
                dimixprop_list[i].x = NULL;dimixprop_list[i].p = pvec[i+1]; 
                dimixprop_list[i].n=0; dimixprop_list[i].tc=1;
            }
        }
    }

    // dinfo for calibration -- used for getting predictions from mixed model
    dinfo dipred;
    dipred.n=0;dipred.p=pvec[0],dipred.x = NULL;dipred.y=NULL;dipred.tc=tc;
    std::vector<double> fp=y_list[0];
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
        dipred.n=nvec[0]; dipred.x = &x_list[0][0]; dipred.y = &fp[0];
#ifdef _OPENMPI
   } 
#endif

    // Calibration vectors and sumwr2
    double csumwr2=0, nsumwr2=0;
    bool hardreject;
    size_t usize = umap.size();
    std::vector<double> unew(usize,0),ucur(usize,0);
    std::vector<size_t> uaccept(usize,0),ureject(usize,0);

    // Start the MCMC
#ifdef _OPENMPI
    double tstart=0.0,tend=0.0;
    if(mpirank==0) tstart=MPI_Wtime();
    if(mpirank==0) cout << "Starting MCMC..." << endl;
#else
    cout << "Starting MCMC..." << endl;
#endif
    // Initialize finfo using predictions from each emulator
    if(mpirank > 0){
        for(int j=0;j<nummodels;j++){
            ambm_list[j]->predict(&dimix_list[j]);
            for(size_t k=0;k<dimix_list[j].n;k++){
                fi(k,j) = fmix_list[j][k] + means_list[j+1];
                //cout << "fi(k,j) = " << fi(k,j) << endl; 
            }   
        }
    }
    // Adapt Stage in the MCMC
    diterator diter0(&dips_list[0]);
    for(size_t i=0;i<nadapt;i++) { 
        // Print adapt step number
        if((i % printevery) ==0 && mpirank==0) cout << "Adapt iteration " << i << endl;
#ifdef _OPENMPI  
        //Emulation Steps
        for(int j=0;j<nummodels;j++){
            // Update emulator
            if(mpirank==0){ambm_list[j]->draw(gen);} else {ambm_list[j]->draw_mpislave(gen);}
            if(mpirank>0){
                //Update finfo column 
                ambm_list[j]->predict(&dimix_list[j]);
                for(size_t l=0;l<dimix_list[j].n;l++){
                    //fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; // f_mix is only K dimensional -- hence using j as its index
                    fi(l,j) = fmix_list[j][l] + means_list[j+1];
                    //cout << "fi(l,j) = " << fi(l,j) << endl; 
                }
            }  
        }

        // Model Mixing step
        if(mpirank==0){axb.drawvec(gen);} else {axb.drawvec_mpislave(gen);}
        
        // Calibration steps
        if(modeltype==MODEL_MIXCALIBRATE){
            // Reset csumwr2 and nsumwr2
            csumwr2 = 0.0; 
            nsumwr2 = 0.0;
            
            // Get current weighted sum of residuals squared (only need field obs)
            for(size_t j=0;j<nvec[0];j++){csumwr2+=(axb.r(j)/sig_vec[0][j])*(axb.r(j)/sig_vec[0][j]);}
            
            // Get joint proposal and update xf_copy with new u
            if(mpirank==0) uvec.drawnew(gen); else uvec.drawnew_mpi(gen);
            
            hardreject = false;
            for(size_t j=0;j<pu;j++){
                if(uvec.unew[j] < avec[j] || uvec.unew[j] > bvec[j]){
                    hardreject = true;
                    //cout << "HARD REJECT" << endl;
                }
            }
            
            // get new predictions
            if(mpirank>0 && !hardreject){
                for(int j=0;j<nummodels;j++){
                    uvec.updatexmm(xf_list[j],u_cols_list[j],pvec[j+1],nvec[0]);
                    //Update finfo column 
                    ambm_list[j]->predict(&dimixprop_list[j]);
                    for(size_t l=0;l<dimixprop_list[j].n;l++){
                        fiprop(l,j) = fmixprop_list[j][l] + means_list[j+1];  
                        //cout << "fiprop(l,j) = " << fiprop(l,j) << endl;                       
                    }
                }
            }
            if(!hardreject){
                // Get mixed predictions, using the same dinfo for the field obs variance model for convenience
                axb.predict_vec(&dips_list[0],&fiprop);

                // Get new weight sum of residuals squared, r_list[0][j] holds the new predictions (for convenience)
                for(size_t j=0;j<nvec[0];j++){nsumwr2+=((y_list[0][j]-r_list[0][j])/sig_vec[0][j])*((y_list[0][j]-r_list[0][j])/sig_vec[0][j]);} 
                
                // Now do the MH Step
                uvec.mhstep(csumwr2,nsumwr2,gen);

                // Update the x data if the propsed move was accepted
                if(uvec.accept){
                    for(int j=0;j<nummodels;j++){
                        uvec.updatexmm(xf_list[j],u_cols_list[j],pvec[j+1],nvec[0]);
                    }
                }   
            }else{
                for(size_t j=0;j<pu;j++) uvec.rejectvec[j]++;
            }
            if((i+1)%adaptevery==0 && mpirank==0) uvec.adapt();
        }

#else
        // Emulation Steps
        for(int j=0;j<nummodels;j++){    
            // Emulation step
            ambm_list[j]->drawvec(gen);
            
            // Update finfo column
            ambm_list[j]->predict(&dimix_list[j]);
            for(int l=0;l<nvec[0];l++){
                //fi(l,j+1) = fmix_list[j][l] + means_list[j+1];
                fi(l,j) = fmix_list[j][l] + means_list[j+1];
            }
        }
        
        // Model Mixing step
        axb.drawvec(gen);
        
#endif
    // Get fitted values and update the residuals for the variance model
#ifdef _OPENMPI
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            if(j>0){
                // Emulators, get fitted values and compute the residuals
                if(mpirank>0){
                    ambm_list[j-1]->predict(&dips_list[j]);
                    for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}    
                }else{
                    dips_list[j] = dinfo_list[j];
                }
                if((i+1)%adaptevery==0 && mpirank==0){ambm_list[j-1]->adapt();}
            }else{
                // Model Mixing
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];}
                dips_list[0] -= faxb;
                if((i+1)%adaptevery==0 && mpirank==0){axb.adapt();}
            }
        }
#else
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            //dips_list[j] = dinfo_list[j];
            if(j>0){
                // Emulators
                ambm_list[j-1]->predict(&dips_list[j]);
                for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}                    
                if((i+1)%adaptevery==0 && mpirank==0){ambm_list[j-1]->adapt();}
            }else{
                // Model Mixing
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];}
                dips_list[0] -= faxb;
                if((i+1)%adaptevery==0 && mpirank==0){axb.adapt();}
            }
        }

#endif

#ifdef _OPENMPI
        // Draw the variances
        // Model Mixing
        if(mpirank==0) pxb.draw(gen); else pxb.draw_mpislave(gen);
        //if(mpirank>0){cout << "sigdraw[0][0] = " << sigmav_list[0][0] << endl;} 

        // Emulators
        for(int j=0;j<nummodels;j++){
            if(mpirank==0) psbm_list[j]->draw(gen); else psbm_list[j]->draw_mpislave(gen);
        }

#else
        // Draw for model mixing
        pxb.draw(gen);        
        // Draw for emulators
        for(int j=0;j<nummodels;j++){psbm_list[j]->draw(gen);}
#endif 
        for(int j=0;j<=nummodels;j++){
            if(j>0){
                // Update the variance for the computer model runs
                psbm_list[j-1]->predict(&disig_list[j]);
                if((i+1)%adaptevery==0 && mpirank==0) psbm_list[j-1]->adapt();
            }else{
                disig_list[0] = fpxb;
                if((i+1)%adaptevery==0 && mpirank==0) pxb.adapt();
            }
        }
    }

    //-----------------------------------------------------
    // Burn-in stage
    //-----------------------------------------------------
    for(size_t i=0;i<nburn;i++) { 
        // Print adapt step number
        if((i % printevery) ==0 && mpirank==0) cout << "Burn iteration " << i << endl;
#ifdef _OPENMPI      
        //Emulation Steps
        for(int j=0;j<nummodels;j++){
            // Update emulator
            if(mpirank==0){ambm_list[j]->draw(gen);} else {ambm_list[j]->draw_mpislave(gen);}
            
            if(mpirank>0){
                //Update finfo column 
                ambm_list[j]->predict(&dimix_list[j]);
                for(size_t l=0;l<dimix_list[j].n;l++){
                    fi(l,j) = fmix_list[j][l] + means_list[j+1];
                    //if(j==0) cout << "fi(l,j) = " << fi(l,j) << endl; 
                }
            }           
        } 
        
        // Model Mixing step
        if(mpirank==0){axb.drawvec(gen);} else {axb.drawvec_mpislave(gen);}

        // Calibration steps
        if(modeltype==MODEL_MIXCALIBRATE){
            // Reset csumwr2 and nsumwr2
            csumwr2 = 0.0; 
            nsumwr2 = 0.0;
            
            // Get current weighted sum of residuals squared (only need field obs)
            for(size_t j=0;j<nvec[0];j++){csumwr2+=(axb.r(j)/sig_vec[0][j])*(axb.r(j)/sig_vec[0][j]);}
            
            // Get joint proposal and update xf_copy with new u
            if(mpirank==0) uvec.drawnew(gen); else uvec.drawnew_mpi(gen);
            
            hardreject = false;
            for(size_t j=0;j<pu;j++){
                if(uvec.unew[j] < avec[j] || uvec.unew[j] > bvec[j]) hardreject = true;
            }

            // get new predictions
            if(mpirank>0 && !hardreject){
                for(int j=0;j<nummodels;j++){
                    uvec.updatexmm(xf_list[j],u_cols_list[j],pvec[j+1],nvec[0]);
                    //Update finfo column 
                    ambm_list[j]->predict(&dimixprop_list[j]);
                    for(size_t l=0;l<dimixprop_list[j].n;l++){
                        //fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; // f_mix is only K dimensional -- hence using j as its index
                        fiprop(l,j) = fmixprop_list[j][l] + means_list[j+1];
                        //cout << "fi(l,j) = " << fi(l,j) << endl; 
                    }
                }
            }
            if(!hardreject){
                // Get mixed predictions, using the same dinfo for the field obs variance model for convenience                
                axb.predict_vec(&dips_list[0],&fiprop);

                // Get new weight sum of residuals squared, r_list[0][j] holds the new predictions (for convenience)
                for(size_t j=0;j<nvec[0];j++){nsumwr2+=((y_list[0][j]-r_list[0][j])/sig_vec[0][j])*((y_list[0][j]-r_list[0][j])/sig_vec[0][j]);} 

                // Now do the MH Step
                uvec.mhstep(csumwr2,nsumwr2,gen);
                // Update the x data if the propsed move was accepted
                if(uvec.accept){
                    for(int j=0;j<nummodels;j++){
                        uvec.updatexmm(xf_list[j],u_cols_list[j],pvec[j+1],nvec[0]);
                    }                
                }   
            }
        }

#else
        // Emulation Steps
        for(int j=0;j<nummodels;j++){    
            // Emulation step
            ambm_list[j]->drawvec(gen);
            
            // Update finfo column
            ambm_list[j]->predict(&dimix_list[j]);
            for(int l=0;l<nvec[0];l++){
                //fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; 
                fi(l,j) = fmix_list[j][l] + means_list[j+1];
            }
        }

        // Model Mixing step
        axb.drawvec(gen);
#endif
    
    // Get fitted values and update the residuals for the variance model
#ifdef _OPENMPI
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            //dips_list[j] = dinfo_list[j];
            if(j>0){
                // Emulators
                if(mpirank>0){
                    ambm_list[j-1]->predict(&dips_list[j]);
                    for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}    
                }else{
                    dips_list[j] = dinfo_list[j];
                }
            }else{
                // Model Mixing
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];}
                dips_list[0] -= faxb;
            }
        }
#else
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            //dips_list[j] = dinfo_list[j];
            if(j>0){
                // Emulators
                ambm_list[j-1]->predict(&dips_list[j]);
                for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}                    
            }else{
                // Model Mixing
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];}
                dips_list[0] -= faxb;
            }
        }

#endif

#ifdef _OPENMPI
        // Draw the variances
        // Model Mixing
        if(mpirank==0) pxb.draw(gen); else pxb.draw_mpislave(gen);

        // Emulators
        for(int j=0;j<nummodels;j++){
            if(mpirank==0) psbm_list[j]->draw(gen); else psbm_list[j]->draw_mpislave(gen);
        }

#else
        // Draw for model mixing
        pxb.draw(gen);        
        // Draw for emulators
        for(int j=0;j<nummodels;j++){psbm_list[j]->draw(gen);}
#endif 
        for(int j=0;j<=nummodels;j++){
            if(j>0){
                // Update the variance for the computer model runs
                psbm_list[j-1]->predict(&disig_list[j]);
            }else{
                disig_list[0] = fpxb;
            }
            
        }
    }

    //-----------------------------------------------------
    // Draw stage
    //-----------------------------------------------------
    // Save the most recent calibration parameter vector
    if(modeltype==MODEL_MIXCALIBRATE){
        udraws.insert(udraws.end(),uvec.ucur.begin(),uvec.ucur.end());
    }

    for(size_t i=0;i<nd;i++) { 
        // Print adapt step number
        if((i % printevery) ==0 && mpirank==0) cout << "Draw iteration " << i << endl;
#ifdef _OPENMPI      
        //Emulation Steps
        for(int j=0;j<nummodels;j++){
            // Update emulator
            if(mpirank==0){ambm_list[j]->draw(gen);} else {ambm_list[j]->draw_mpislave(gen);}
            
            if(mpirank>0){
                //Update finfo column 
                ambm_list[j]->predict(&dimix_list[j]);
                for(size_t l=0;l<dimix_list[j].n;l++){
                    //fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; // f_mix is only K dimensional -- hence using j as its index
                    fi(l,j) = fmix_list[j][l] + means_list[j+1];
                    //cout << "fi(l,j+1) = " << fi(l,j+1) << endl; 
                }
            }           
        } 
        
        // Model Mixing step
        if(mpirank==0){axb.drawvec(gen);} else {axb.drawvec_mpislave(gen);}

        // Calibration steps
        if(modeltype==MODEL_MIXCALIBRATE){
             // Reset csumwr2 and nsumwr2
            csumwr2 = 0.0; 
            nsumwr2 = 0.0;
            
            // Get current weighted sum of residuals squared (only need field obs)
            for(size_t j=0;j<nvec[0];j++){csumwr2+=(axb.r(j)/sig_vec[0][j])*(axb.r(j)/sig_vec[0][j]);}
            
            // Get joint proposal and update xf_copy with new u
            if(mpirank==0) uvec.drawnew(gen); else uvec.drawnew_mpi(gen);
            //cout << "unew = " << uvec.unew[0] << endl;
            hardreject = false;
            for(size_t j=0;j<pu;j++){
                if(uvec.unew[j] < avec[j] || uvec.unew[j] > bvec[j]) hardreject = true;
            }

            // get new predictions
            if(mpirank>0 && !hardreject){
                for(int j=0;j<nummodels;j++){
                    uvec.updatexmm(xf_list[j],u_cols_list[j],pvec[j+1],nvec[0]);
                    //Update finfo column 
                    ambm_list[j]->predict(&dimixprop_list[j]);
                    for(size_t l=0;l<dimixprop_list[j].n;l++){
                        //fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; // f_mix is only K dimensional -- hence using j as its index
                        fiprop(l,j) = fmixprop_list[j][l] + means_list[j+1];
                        //cout << "fi(l,j+1) = " << fi(l,j+1) << endl; 
                    }
                }
            }
            if(!hardreject){    
                // Get mixed predictions, using the same dinfo for the field obs variance model for convenience
                axb.predict_vec(&dips_list[0],&fiprop);
            
                // Get new weight sum of residuals squared, r_list[0][j] holds the new predictions (for convenience)
                for(size_t j=0;j<nvec[0];j++){nsumwr2+=((y_list[0][j]-r_list[0][j])/sig_vec[0][j])*((y_list[0][j]-r_list[0][j])/sig_vec[0][j]);} 

                // Now do the MH Step
                uvec.mhstep(csumwr2,nsumwr2,gen);
                // Update the x data if the propsed move was accepted
                if(uvec.accept){
                    for(int j=0;j<nummodels;j++){
                        uvec.updatexmm(xf_list[j],u_cols_list[j],pvec[j+1],nvec[0]);
                    }
                }   
            }
        }

#else
        // Emulation Steps
        for(int j=0;j<nummodels;j++){    
            // Emulation step
            ambm_list[j]->drawvec(gen);
            
            // Update finfo column
            ambm_list[j]->predict(&dimix_list[j]);
            for(int l=0;l<nvec[0];l++){
                //fi(l,j+1) = fmix_list[j][l] + means_list[j+1];
                fi(l,j) = fmix_list[j][l] + means_list[j+1]; 
            }
        }

        // Model Mixing step
        axb.drawvec(gen);
#endif
    
    // Get fitted values and update the residuals for the variance model
#ifdef _OPENMPI
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            //dips_list[j] = dinfo_list[j];
            if(j>0){
                // Emulators
                if(mpirank>0){
                    ambm_list[j-1]->predict(&dips_list[j]);
                    for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}    
                }else{
                    dips_list[j] = dinfo_list[j];
                }
            }else{
                // Model Mixing
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];}
                dips_list[0] -= faxb;
            }
        }
#else
        // Set dinfo objecs for the variance
        for(int j=0;j<=nummodels;j++){
            //dips_list[j] = dinfo_list[j];
            if(j>0){
                // Emulators
                ambm_list[j-1]->predict(&dips_list[j]);
                for(size_t l=0;l<dips_list[j].n;l++){r_list[j][l] = y_list[j][l] - r_list[j][l];}                    
            }else{
                // Model Mixing
                for(size_t l=0;l<dips_list[0].n;l++){r_list[0][l] = y_list[0][l];}
                dips_list[0] -= faxb;
            }
        }

#endif

#ifdef _OPENMPI
        // Draw the variances
        // Model Mixing
        if(mpirank==0) pxb.draw(gen); else pxb.draw_mpislave(gen);

        // Emulators
        for(int j=0;j<nummodels;j++){
            if(mpirank==0) psbm_list[j]->draw(gen); else psbm_list[j]->draw_mpislave(gen);
        }

#else
        // Draw for model mixing
        pxb.draw(gen);        
        // Draw for emulators
        for(int j=0;j<nummodels;j++){psbm_list[j]->draw(gen);}
#endif 
        for(int j=0;j<=nummodels;j++){
            if(j>0){
                // Update the variance for the computer model runs
                psbm_list[j-1]->predict(&disig_list[j]);
            }else{
                disig_list[0] = fpxb;
            }
            
        }

        // Save Tree to vector format
        if(mpirank==0) {
            //axb.pr_vec();
            axb.savetree_vec(i,m_list[0],onn_list[0],oid_list[0],ovar_list[0],oc_list[0],otheta_list[0]); 
            pxb.savetree(i,mh_list[0],snn_list[0],sid_list[0],svar_list[0],sc_list[0],stheta_list[0]);
            for(int j=1;j<=nummodels;j++){
                ambm_list[j-1]->savetree(i,m_list[j],onn_list[j],oid_list[j],ovar_list[j],oc_list[j],otheta_list[j]);
                psbm_list[j-1]->savetree(i,mh_list[j],snn_list[j],sid_list[j],svar_list[j],sc_list[j],stheta_list[j]); 
            }
            if(modeltype==MODEL_MIXCALIBRATE){
                udraws.insert(udraws.end(),uvec.ucur.begin(),uvec.ucur.end());
            }
        }
    }
    // Writing data to output files
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
        // Instantiate containers
        std::vector<std::vector<int>*> e_ots(nummodels+1); //=new std::vector<int>(nd*m);
        std::vector<std::vector<int>*> e_oid(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<int>*> e_ovar(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<int>*> e_oc(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<double>*> e_otheta(nummodels+1); //=new std::vector<double>;
        std::vector<std::vector<int>*> e_sts(nummodels+1); //=new std::vector<int>(nd*mh);
        std::vector<std::vector<int>*> e_sid(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<int>*> e_svar(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<int>*> e_sc(nummodels+1); //=new std::vector<int>;
        std::vector<std::vector<double>*> e_stheta(nummodels+1); //=new std::vector<double>;

        // Initialize containers with pointers
        for(int j=0;j<=nummodels;j++){
            e_ots[j]=new std::vector<int>(nd*m_list[j]);
            e_oid[j]=new std::vector<int>;
            e_ovar[j]=new std::vector<int>;
            e_oc[j]=new std::vector<int>;
            e_otheta[j]=new std::vector<double>;
            e_sts[j]=new std::vector<int>(nd*mh_list[j]);
            e_sid[j]=new std::vector<int>;
            e_svar[j]=new std::vector<int>;
            e_sc[j]=new std::vector<int>;
            e_stheta[j]=new std::vector<double>;
        }

        // Loop through each model and store in appropriate outfile 
        for(size_t i=0;i<nd;i++)
            for(int l=0;l<=nummodels;l++) {
                m = m_list[l];
                for(size_t j=0;j<m;j++){ 
                    e_ots[l]->at(i*m+j)=static_cast<int>(oid_list[l][i*m+j].size());
                    e_oid[l]->insert(e_oid[l]->end(),oid_list[l][i*m+j].begin(),oid_list[l][i*m+j].end());
                    e_ovar[l]->insert(e_ovar[l]->end(),ovar_list[l][i*m+j].begin(),ovar_list[l][i*m+j].end());
                    e_oc[l]->insert(e_oc[l]->end(),oc_list[l][i*m+j].begin(),oc_list[l][i*m+j].end());
                    e_otheta[l]->insert(e_otheta[l]->end(),otheta_list[l][i*m+j].begin(),otheta_list[l][i*m+j].end());
                }
            }
        for(size_t i=0;i<nd;i++)
            for(int l=0;l<=nummodels;l++) {
                mh = mh_list[l];
                for(size_t j=0;j<mh;j++){
                    e_sts[l]->at(i*mh+j)=static_cast<int>(sid_list[l][i*mh+j].size());
                    e_sid[l]->insert(e_sid[l]->end(),sid_list[l][i*mh+j].begin(),sid_list[l][i*mh+j].end());
                    e_svar[l]->insert(e_svar[l]->end(),svar_list[l][i*mh+j].begin(),svar_list[l][i*mh+j].end());
                    e_sc[l]->insert(e_sc[l]->end(),sc_list[l][i*mh+j].begin(),sc_list[l][i*mh+j].end());
                    e_stheta[l]->insert(e_stheta[l]->end(),stheta_list[l][i*mh+j].begin(),stheta_list[l][i*mh+j].end());
                }
            }

        //write out to file
        std::ofstream omf;
        std::string ofile;
        //std::ofstream omf_mix(folder + modelname + ".fitmix");
        //std::ofstream omf_emu(folder + modelname + ".fitemu");
        
        for(int j=0;j<=nummodels;j++){
            // Open the mixing for emulation file
            if(j == 0){
                ofile = folder + modelname + ".fitmix";
                omf.open(ofile);
                cout << "\nSaving mixing trees..." << endl;
            }else if(j == 1){
                ofile = folder + modelname + ".fitemulate";
                omf.open(ofile); //opened at first emulator -- kept open until very end
                cout << "Saving emulation trees..." << endl;
            }

            omf << nd << endl;
            omf << m_list[j] << endl;
            omf << mh_list[j] << endl;
            omf << e_ots[j]->size() << endl;
            for(size_t i=0;i<e_ots[j]->size();i++) omf << e_ots[j]->at(i) << endl;
            omf << e_oid[j]->size() << endl;
            for(size_t i=0;i<e_oid[j]->size();i++) omf << e_oid[j]->at(i) << endl;
            omf << e_ovar[j]->size() << endl;
            for(size_t i=0;i<e_ovar[j]->size();i++) omf << e_ovar[j]->at(i) << endl;
            omf << e_oc[j]->size() << endl;
            for(size_t i=0;i<e_oc[j]->size();i++) omf << e_oc[j]->at(i) << endl;
            omf << e_otheta[j]->size() << endl;
            for(size_t i=0;i<e_otheta[j]->size();i++) omf << std::scientific << e_otheta[j]->at(i) << endl;
            omf << e_sts[j]->size() << endl;
            for(size_t i=0;i<e_sts[j]->size();i++) omf << e_sts[j]->at(i) << endl;
            omf << e_sid[j]->size() << endl;
            for(size_t i=0;i<e_sid[j]->size();i++) omf << e_sid[j]->at(i) << endl;
            omf << e_svar[j]->size() << endl;
            for(size_t i=0;i<e_svar[j]->size();i++) omf << e_svar[j]->at(i) << endl;
            omf << e_sc[j]->size() << endl;
            for(size_t i=0;i<e_sc[j]->size();i++) omf << e_sc[j]->at(i) << endl;
            omf << e_stheta[j]->size() << endl;
            for(size_t i=0;i<e_stheta[j]->size();i++) omf << std::scientific << e_stheta[j]->at(i) << endl;
            
            // Close the mixing file before saving the emulation trees
            if(j == 0){
                omf.close();
            }
        }
        // Close the emulation text file
        omf.close();
        
        if(modeltype == MODEL_MIXCALIBRATE){
            //Write calibration parameter -- files
            cout << "Saving calibration parameters..." << endl;
            std::ofstream ouf(folder + modelname + ".udraws");
            for(size_t i=0;i<udraws.size();i++) ouf << udraws.at(i) << endl;
            ouf.close();
        }
        cout << " done." << endl;
    }
    //-------------------------------------------------- 
    // Cleanup.
#ifdef _OPENMPI
    //delete[] lwr_vec; //make pointer friendly
    //delete[] upr_vec; //make pointer friendly
    MPI_Finalize();
#endif
    return 0;
}