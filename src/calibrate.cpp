// Command line interface for calibration models
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
#include "mcbrt.h"
#include "amcbrt.h"
#include "parameters.h"
#include "mbrt.h"
#include "ambrt.h"


using std::cout;
using std::endl;

#define MODEL_OSBART 1
#define MODEL_ORTHBART 2
#define MODEL_MODBART 3
#define MODEL_OSBARTXT 4

int main(int argc, char* argv[])
{
    std::string folder("");

    if(argc>1){
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

    // core filenames for x,y,s,chgv,f
    std::string xcore,ycore,score,chgvcore,fcore;
    conf >> xcore;
    conf >> ycore;
    conf >> score;
    conf >> chgvcore;
    conf >> fcore;
    
    // Means 
    double yfmean, ycmean;
    conf >> yfmean;
    conf >> ycmean;

    //number of trees
    size_t m, mh, me;
    conf >> m;
    conf >> mh;
    conf >> me;

    //nd and burn
    size_t nd, burn, nadapt, adaptevery;
    conf >> nd;
    conf >> burn;
    conf >> nadapt;
    conf >> adaptevery;

    //Priors 
    double mu1,tau1,mu2,tau2;
    conf >> mu1;
    conf >> tau1;
    conf >> mu2;
    conf >> tau2;

    double overalllambdaf, overalllambdac, overallnuf, overallnuc;
    conf >> overalllambdaf;
    conf >> overalllambdac;
    conf >> overallnuf;
    conf >> overallnuc;

    double base, power, baseh, powerh;
    conf >> base;
    conf >> power;
    conf >> baseh;
    conf >> powerh;

    // Calibration parameter info
    size_t pu, tempucol;
    std::vector<size_t> ucols;
    std::vector<std::string> uprior;
    std::vector<std::string> uprop;
    std::vector<double> uparam1, uparam2;
    std::string upr0;
    double up1, up2;
    
    // Column indexes
    conf >> pu;
    for(size_t i=0;i<pu;i++){
        conf >> tempucol;
        ucols.push_back(tempucol);
    }

    // Priors and proposals
    std::vector<double> propwidth(pu,0.25), u0(pu,0);
    std::string proptype;
    conf >> proptype;
    for(size_t i=0;i<pu;i++){
        conf >> upr0;
        conf >> up1;
        conf >> up2;
        uprior.push_back(upr0);
        uparam1.push_back(up1);
        uparam2.push_back(up2);

        if(upr0 == "normal"){
            u0[i] = up1; // init at prior mean
            if(proptype=="default") propwidth[i] = 0.25*4.0*up2; // 25% of +/- 2sd width
        }else if(upr0 == "uniform"){
            u0[i] = (up2+up1)/2; // init at prior mean
            if(proptype=="default") {propwidth[i] = 0.25*(up2-up1);} // 25% of range 
        }
    }

    // Set proposal distributions
    if(proptype=="default"){
        uprop = uprior;
    }else if(proptype=="mala"){
        uprop.resize(pu,"mala");
        cout << "start prop width = " << propwidth[0] << endl;
        propwidth.clear();
        propwidth.resize(pu,2.4/sqrt(pu));
        cout << "new prop width = " << propwidth[0] << endl;
    }

    // Gradient stepsize -- used for finite differences in gradient approx
    std::vector<double> gradstep;
    double gradsz;
    for(size_t i=0;i<pu;i++){
        conf >> gradsz;
        gradstep.push_back(gradsz);
    }
    
    // Control parameters
    //thread count
    int tc;
    conf >> tc;

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
        cout << "OpenBT command-line interface (cli)" << endl;
        cout << "Loading config file at " << folder << endl;
    }

   //--------------------------------------------------
   //read in y -- field obs then model runs
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
        std::ifstream yfile(yfs);
        while(yfile >> ytemp)
            y.push_back(ytemp);
        n=y.size();
#ifndef SILENT
        cout << "node " << mpirank << " loaded " << n << " from " << yfs <<endl;
#endif
#ifdef _OPENMPI
    }
#endif

   //--------------------------------------------------
   //read in x -- inputs for field obs then model runs
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
        std::ifstream xfile(xfs);
        while(xfile >> xtemp)
            x.push_back(xtemp);
        p = x.size()/n;
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
    size_t nf=0, nc=0;
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

    // compute nf and nc
    nf = fi.colwise().sum()(1);
    nc = n - nf;

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
    /*
    diterator diter(&di);
    for(;diter<diter.until();diter++){
        cout << "gety = " << diter.gety() << endl;
    }
    */
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
    for(size_t i=0;i<p;i++){
        cout << "Variable " << i << " has numcuts=" << xi[i].size() << " : ";
        cout << xi[i][0] << " ... " << xi[i][xi[i].size()-1] << endl;
    }
#endif

    //--------------------------------------------------
    // Setup model objects
    //--------------------------------------------------
    amcbrt acb(m);

    //cutpoints
    acb.setxi(&xi);    //set the cutpoints for this model object
    //function output information
    acb.setfi(&fi, k);
    //data objects
    acb.setdata_vec(&di);  //set the data
    //thread count
    acb.settc(tc-1);      //set the number of slaves when using MPI.
    //mpi rank
    #ifdef _OPENMPI
    acb.setmpirank(mpirank);  //set the rank when using MPI.
    acb.setmpicvrange(lwr,upr); //range of variables each slave node will update in MPI change-of-var proposals.
    #endif
    //tree prior
    acb.settp(base,power);
    //MCMC info
    acb.setmi(
            pbd,  //probability of birth/death
            pb,  //probability of birth
            minnumbot,    //minimum number of observations in a bottom node
            dopert, //do perturb/change variable proposal?
            stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv  //initialize the change of variable correlation matrix.
            );
    
    acb.setci(mu1,mu2,tau1,tau2,sig);
    acb.setuvars(ucols); // set calibration vector
    acb.setetatreeinfo(me); // set eta only trees

    if(modeltype == MODEL_ORTHBART){acb.set_orthogonal_delta(true);}else{acb.set_orthogonal_delta(false);}
    if(modeltype == MODEL_MODBART){acb.set_modularization(true);}else{acb.set_modularization(false);}

    //--------------------------------------------------
    // Set product variance models
    psbrt psbmf(mh);

    //make di for psbrt object
    dinfo dipsf;
    dipsf.n=0; dipsf.p=p; dipsf.x=NULL; dipsf.y=NULL; dipsf.tc=tc;
    double *rf=NULL;
    std::vector<double> xf, xftemp, yf;
#ifdef _OPENMPI
    if(mpirank>0) {
#endif
    rf = new double[nf]; // change to nf
    size_t j = 0;
    for(size_t i=0;i<n;i++){
        // Is this a field obs
        if(fi(i,1)==1){
            rf[j]=sigmav[i];
            xftemp.clear();
            xftemp = {x.begin()+p*i,x.begin()+p*(i+1)};
            //cout << "xftemp[0] = " << xftemp[0] << ".... xftemp[p-1] = " << xftemp[p-1] << "---" << endl;
            xf.insert(xf.end(),xftemp.begin(),xftemp.end());
            yf.push_back(y[i]);
            j = j + 1;
        }
    } 
    dipsf.x=&xf[0]; dipsf.y=rf; dipsf.n=nf;
#ifdef _OPENMPI
    }
#endif

    double opmf=1.0/((double)mh);
    double nuf=2.0*pow(overallnuf,opmf)/(pow(overallnuf,opmf)-pow(overallnuf-2.0,opmf));
    double lambdaf=pow(overalllambdaf,opmf);

    //cutpoints
    psbmf.setxi(&xi);    
    //data objects
    psbmf.setdata(&dipsf);  
    //thread count
    psbmf.settc(tc-1); 
    //mpi rank
#ifdef _OPENMPI
    psbmf.setmpirank(mpirank);  //set the rank when using MPI.
    psbmf.setmpicvrange(lwr,upr); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
    //tree prior, MCMC info, and conditioning info
    psbmf.settp(baseh,powerh);
    psbmf.setmi(pbdh,pbh,minnumboth,doperth,stepwperth,probchvh,&chgv);
    psbmf.setci(nuf,lambdaf);


    //--------------------------------------------------
    // Product variance model for the error in the model runs
    psbrt psbmc(mh);

    //make di for psbrt object -- model runs
    dinfo dipsc;
    dipsc.n=0; dipsc.p=p; dipsc.x=NULL; dipsc.y=NULL; dipsc.tc=tc;
    double *rc=NULL;
    std::vector<double> xc, xctemp, yc;
#ifdef _OPENMPI
    if(mpirank>0) {
#endif
    rc = new double[nc];
    size_t j = 0;
    for(size_t i=0;i<n;i++){
        // Check, is this a model run
        if(fi(i,1) == 0){
            rc[j]=sigmav[i];
            xctemp.clear();
            xctemp = {x.begin()+p*i,x.begin()+p*(i+1)};
            //cout << "xctemp[0] = " << xctemp[0] << ".... xctemp[p-1] = " << xctemp[p-1] << "---" << endl;
            xc.insert(xc.end(),xctemp.begin(),xctemp.end());
            yc.push_back(y[i]);
            j = j + 1;
        }
    } 
    dipsc.x=&xc[0]; dipsc.y=rc; dipsc.n=nc;
#ifdef _OPENMPI
   }
#endif

    double opmc=1.0/((double)mh);
    double nuc=2.0*pow(overallnuc,opmc)/(pow(overallnuc,opmc)-pow(overallnuc-2.0,opmc));
    double lambdac=pow(overalllambdac,opmc);

    //cutpoints
    psbmc.setxi(&xi);    //set the cutpoints for this model object
    //data objects
    psbmc.setdata(&dipsc);  //set the data
    //thread count
    psbmc.settc(tc-1); 
    //mpi rank
#ifdef _OPENMPI
    psbmc.setmpirank(mpirank);  //set the rank when using MPI.
    psbmc.setmpicvrange(lwr,upr); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
    //tree prior, MCMC info, and conditioning info
    psbmc.settp(baseh,powerh);
    psbmc.setmi(pbdh,pbh,minnumboth,doperth,stepwperth,probchvh,&chgv);
    psbmc.setci(nuc,lambdac);


    //--------------------------------------------------
    // Set calibration parameters
    param uvec(pu);
    uvec.settc(tc-1);
    uvec.setmpirank(mpirank);
    uvec.setpriors(uprior,uparam1,uparam2);
    uvec.setproposals(uprop,propwidth);
    uvec.setucur(u0);

    //--------------------------------------------------
    // Setup mcmc containers
    //--------------------------------------------------
    // Calibration model
    std::vector<int> onn(nd*m,1);
    std::vector<std::vector<int> > oid(nd*m, std::vector<int>(1));
    std::vector<std::vector<int> > ovar(nd*m, std::vector<int>(1));
    std::vector<std::vector<int> > oc(nd*m, std::vector<int>(1));
    std::vector<std::vector<double> > otheta(nd*m, std::vector<double>(1));

    // Variance model for field obs
    std::vector<int> sfnn(nd*mh,1);
    std::vector<std::vector<int> > sfid(nd*mh, std::vector<int>(1));
    std::vector<std::vector<int> > sfvar(nd*mh, std::vector<int>(1));
    std::vector<std::vector<int> > sfc(nd*mh, std::vector<int>(1));
    std::vector<std::vector<double> > sftheta(nd*mh, std::vector<double>(1));

    // Variance model for model runs
    std::vector<int> scnn(nd*mh,1);
    std::vector<std::vector<int> > scid(nd*mh, std::vector<int>(1));
    std::vector<std::vector<int> > scvar(nd*mh, std::vector<int>(1));
    std::vector<std::vector<int> > scc(nd*mh, std::vector<int>(1));
    std::vector<std::vector<double> > sctheta(nd*mh, std::vector<double>(1));

    // Wrappers
    //brtMethodWrapper facb(&brt::f,acb);
    //brtMethodWrapper fpsbmf(&brt::f,psbmf);
    //brtMethodWrapper fpsbmc(&brt::f,psbmc);

    // Parameters
    std::vector<double> udraws;
    double csumwr2, nsumwr2; // current and new wtd residuals squared
    double cprop, nprop; // proposal probabilities cprop = c->n, nprop = n->c
    std::vector<double> xf_copy = xf; // used for predictions in proposal
    std::vector<double> xf_grad = xf; // used for gradient calculations in proposal
    std::vector<double> grad;
    
    // dinfo for the calibration predictions during proposal
    dinfo di_prop;
    double *fprop = new double[nf];
    di_prop.n=0;di_prop.p=p,di_prop.x = NULL;di_prop.y=NULL;di_prop.tc=tc;
    mxd fif; // design matrix for field obs, its nf x 2 and all ones! 
    fif = mxd::Ones(nf,2);
#ifdef _OPENMPI
    if(mpirank>0) { 
#endif
        di_prop.n=nf; di_prop.x = &xf_copy[0]; di_prop.y = &fprop[0]; 
#ifdef _OPENMPI
    }
#endif

    //--------------------------------------------------
    // MCMC Time
    //--------------------------------------------------
#ifdef _OPENMPI
    double tstart=0.0,tend=0.0;
    if(mpirank==0) tstart=MPI_Wtime();
    if(mpirank==0) cout << "Starting MCMC..." << endl;
#else
    cout << "Starting MCMC..." << endl;
#endif
    //------------------------------------------------------------------------------
    // Adapt stage
    //------------------------------------------------------------------------------
    for(size_t i=0;i<nadapt;i++) { 
        if((i % printevery) ==0 && mpirank==0) cout << "Adapt iteration " << i << endl;
        // Update mean trees
#ifdef _OPENMPI
        if(mpirank==0){acb.drawvec(gen);} else {acb.drawvec_mpislave(gen);}
#else
        acb.drawvec(gen);
#endif
        // Update the residuals, not using methodwrappper now bc I didn't create additional dinfo objects 
        for(size_t j=0;j<nf;j++){
            rf[j] = y[j]-acb.f(j);
            //cout << "acb.f(j) + ycmean = " << acb.f(j) + ycmean << endl;    
        }
        for(size_t j=0;j<nc;j++){
            rc[j] = y[nf+j]-acb.f(nf+j);
            //cout << "acb.f(nf+j) = " << acb.f(nf+j) << endl;
        }
        
        if((i+1)%adaptevery==0 && mpirank==0) acb.adapt();
        // Update varaince trees
#ifdef _OPENMPI
        if(mpirank==0) psbmf.draw(gen); else psbmf.draw_mpislave(gen);
        if(mpirank==0) psbmc.draw(gen); else psbmc.draw_mpislave(gen);
#else
        psbmf.draw(gen);
        psbmc.draw(gen);
#endif
        // Update the value of sigma in sig
        for(size_t j=0;j<nf;j++){
            sig[j] = psbmf.f(j);
            //cout << "sig[j] = " << sig[j] << endl;    
        }
        for(size_t j=0;j<nc;j++){
            sig[nf+j] = psbmc.f(j);
            //cout << "sig[nf+j] = " << sig[nf+j] << endl;    
        }
        if((i+1)%adaptevery==0 && mpirank==0) psbmf.adapt();
        if((i+1)%adaptevery==0 && mpirank==0) psbmc.adapt();

        // Update calibration parameters
#ifdef _OPENMPI
        // Reset csumwr2 and nsumwr2
        csumwr2 = 0.0; nsumwr2 = 0.0;
        cprop = 0.0; nprop = 0.0;

        // Get current weighted sum of residuals squared (only need field obs)
        for(size_t j=0;j<nf;j++){csumwr2+=(acb.r(j)/sig[j])*(acb.r(j)/sig[j]);}
        
        // Get joint proposal and update xf_copy with new u -- clean up into one draw function (??)
        if(proptype=="default"){
            if(mpirank==0) uvec.drawnew(gen); else uvec.drawnew_mpi(gen);
        }else if(proptype=="mala"){
            // Get gradient, proposed move, and proposal prob
            xf_grad = xf; // update xf_grad
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.ucur);
            //cout << "grad = " << grad[0] << endl;
            if(mpirank==0) uvec.drawnew_mala(grad,gen); else uvec.drawnew_mpi(gen); //mpi function is the exact same
            if(mpirank==0) cprop = uvec.logprp_mala(uvec.ucur, uvec.unew, grad);
            //cout << "cprop = " << cprop << endl;
        }
        // Update x after the proposal       
        if(mpirank>0) uvec.updatex(xf_copy,ucols,p,nf);

        // Get predictions for field obs with new u
        if(mpirank>0) acb.predict_vec(&di_prop,&fif);
        
        // Get new weight sum of residuals squared
        for(size_t j=0;j<nf;j++){nsumwr2+=((yf[j]-fprop[j])/sig[j])*((yf[j]-fprop[j])/sig[j]);}

        if(proptype == "mala"){
            // get gradient for proposal
            xf_grad = xf_copy;
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.unew);
            if(mpirank>0) {nprop = uvec.logprp_mala(uvec.unew, uvec.ucur, grad);}

            // Update csumwr2 and nsumwr2 to include proposal effect
            csumwr2 += cprop;
            nsumwr2 += nprop;
        }
        // Now do the MH Step
        uvec.mhstep(csumwr2,nsumwr2,gen);
        // Update the x data if the propsed move was accepted
        if(uvec.accept){
            uvec.updatex(x,ucols,p,nf); // used in the mean model
            uvec.updatex(xf,ucols,p,nf); // used in the f variance model
        }

#else
        // Reset csumwr2 and nsumwr2
        csumwr2 = 0.0; 
        nsumwr2 = 0.0;
        cprop = 0.0; nprop = 0.0;

        // Get current weighted sum of residuals squared (only need field obs)
        for(size_t j=0;j<nf;j++){csumwr2+=(acb.r(j)/sig[j])*(acb.r(j)/sig[j]);}
        // Get joint proposal and update xf_copy with new u
        if(proptype=="default"){
            uvec.drawnew(gen);
        }else if(proptype=="mala"){
            // Get gradient, proposed move, and proposal prob
            xf_grad = xf; // update xf_grad
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.ucur);
            uvec.drawnew_mala(grad,gen); 
            cprop = uvec.logprp_mala(uvec.ucur, uvec.unew, grad);
        }
        uvec.updatex(xf_copy,ucols,p,nf);
        // Get predictions for field obs with new u
        acb.predict_vec(&di_prop,&fif);
        // Get new weight sum of residuals squared
        for(size_t j=0;j<nf;j++){nsumwr2+=((yf[j]-fprop[j])/sig[j])*((yf[j]-fprop[j])/sig[j]);}
        // MALA proposal probabilities
        if(proptype == "mala"){
            // get gradient for proposal
            xf_grad = xf_copy;
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.unew);
            nprop = uvec.logprp_mala(uvec.unew, uvec.ucur, grad);

            // Update csumwr2 and nsumwr2 to include proposal effect
            csumwr2 += cprop;
            nsumwr2 += nprop;
        }
        // Now do the MH Step
        uvec.mhstep(csumwr2,nsumwr2,gen);

        // Update the x data if the propsed move was accepted
        if(uvec.accept){
            uvec.updatex(x,ucols,p,nf); // used in the mean model
            uvec.updatex(xf,ucols,p,nf); // used in the f variance model
        }
        
#endif 
        if((i+1)%adaptevery==0 && mpirank==0) uvec.adapt();
    }

    //------------------------------------------------------------------------------
    // Burn-in stage
    //------------------------------------------------------------------------------

    for(size_t i=0;i<burn;i++) { 
        if((i % printevery) ==0 && mpirank==0) cout << "Burn iteration " << i << endl;
        // Update mean trees
#ifdef _OPENMPI
        if(mpirank==0){acb.drawvec(gen);} else {acb.drawvec_mpislave(gen);}
#else
        acb.drawvec(gen);
#endif
        // Update the residuals, not using methodwrappper now bc I didn't create additional dinfo objects 
        for(size_t j=0;j<nf;j++){
            rf[j] = y[j]-acb.f(j);    
        }
        for(size_t j=0;j<nc;j++){
            rc[j] = y[nf+j]-acb.f(nf+j);    
        }
        // Update varaince trees
#ifdef _OPENMPI
        if(mpirank==0) psbmf.draw(gen); else psbmf.draw_mpislave(gen);
        if(mpirank==0) psbmc.draw(gen); else psbmc.draw_mpislave(gen);
#else
        psbmf.draw(gen);
        psbmc.draw(gen);
#endif
        // Update the value of sigma in sig
        for(size_t j=0;j<nf;j++){
            sig[j] = psbmf.f(j);    
        }
        for(size_t j=0;j<nc;j++){
            sig[nf+j] = psbmc.f(j);    
        }
        
        // Update calibration parameters
#ifdef _OPENMPI
        // Reset csumwr2 and nsumwr2
        csumwr2 = 0.0; nsumwr2 = 0.0;
        cprop = 0.0; nprop = 0.0;
        // Get current weighted sum of residuals squared (only need field obs)
        for(size_t j=0;j<nf;j++){csumwr2+=(acb.r(j)/sig[j])*(acb.r(j)/sig[j]);}

        // Get joint proposal and update xf_copy with new u
        if(proptype=="default"){
            if(mpirank==0) uvec.drawnew(gen); else uvec.drawnew_mpi(gen);
        }else if(proptype=="mala"){
            // Get gradient, proposed move, and proposal prob
            xf_grad = xf; // update xf_grad
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.ucur);
            if(mpirank==0) uvec.drawnew_mala(grad,gen); else uvec.drawnew_mpi(gen); //mpi function is the exact same
            if(mpirank==0) cprop = uvec.logprp_mala(uvec.ucur, uvec.unew, grad);
        }
        if(mpirank>0) uvec.updatex(xf_copy,ucols,p,nf);

        // Get predictions for field obs with new u
        if(mpirank>0) acb.predict_vec(&di_prop,&fif);

        // Get new weight sum of residuals squared
        for(size_t j=0;j<nf;j++){nsumwr2+=((yf[j]-fprop[j])/sig[j])*((yf[j]-fprop[j])/sig[j]);}
        if(proptype == "mala"){
            // get gradient for proposal
            xf_grad = xf_copy;
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.unew);
            if(mpirank>0) {nprop = uvec.logprp_mala(uvec.unew, uvec.ucur, grad);}

            // Update csumwr2 and nsumwr2 to include proposal effect
            csumwr2 += cprop;
            nsumwr2 += nprop;
        }
        // Now do the MH Step
        uvec.mhstep(csumwr2,nsumwr2,gen);

        // Update the x data if the propsed move was accepted
        if(uvec.accept){
            uvec.updatex(x,ucols,p,nf); // used in the mean model
            uvec.updatex(xf,ucols,p,nf); // used in the f variance model
        } 

#else
        // Reset csumwr2 and nsumwr2
        csumwr2 = 0.0; 
        nsumwr2 = 0.0;
        cprop = 0.0; nprop = 0.0;
        // Get current weighted sum of residuals squared (only need field obs)
        for(size_t j=0;j<nf;j++){csumwr2+=(acb.r(j)/sig[j])*(acb.r(j)/sig[j]);}
        // Get joint proposal and update xf_copy with new u
        if(proptype=="default"){
            uvec.drawnew(gen);
        }else if(proptype=="mala"){
            // Get gradient, proposed move, and proposal prob
            xf_grad = xf; // update xf_grad
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.ucur);
            uvec.drawnew_mala(grad,gen); 
            cprop = uvec.logprp_mala(uvec.ucur, uvec.unew, grad);
        }
        uvec.updatex(xf_copy,ucols,p,nf);
        
        // Get predictions for field obs with new u
        acb.predict_vec(&di_prop,&fif);
        // Get new weight sum of residuals squared
        for(size_t j=0;j<nf;j++){nsumwr2+=((yf[j]-fprop[j])/sig[j])*((yf[j]-fprop[j])/sig[j]);}
        // MALA proposal probabilities
        if(proptype == "mala"){
            // get gradient for proposal
            xf_grad = xf_copy;
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.unew);
            nprop = uvec.logprp_mala(uvec.unew, uvec.ucur, grad);

            // Update csumwr2 and nsumwr2 to include proposal effect
            csumwr2 += cprop;
            nsumwr2 += nprop;
        }
        // Now do the MH Step
        uvec.mhstep(csumwr2,nsumwr2,gen);

        // Update the x data if the propsed move was accepted
        if(uvec.accept){
            uvec.updatex(x,ucols,p,nf); // used in the mean model
            uvec.updatex(xf,ucols,p,nf); // used in the f variance model
        }
#endif 
    }

    //------------------------------------------------------------------------------
    // Draw stage
    //------------------------------------------------------------------------------
    // Save the last burn-in draw from u -- this is used in prediction to initialize the u values
    udraws.insert(udraws.end(),uvec.ucur.begin(),uvec.ucur.end());
    for(size_t i=0;i<nd;i++){ 
        if((i % printevery) ==0 && mpirank==0) cout << "Draw iteration " << i << endl;
        // Update mean trees
#ifdef _OPENMPI
        if(mpirank==0){acb.drawvec(gen);} else {acb.drawvec_mpislave(gen);}
#else
        acb.drawvec(gen);
#endif
        // Update the residuals, not using methodwrappper now bc I didn't create additional dinfo objects 
        for(size_t j=0;j<nf;j++){
            rf[j] = y[j]-acb.f(j);
            //cout << "y[j] = " << y[j] << "--- acb.f(j) = " << acb.f(j) << "----rf[j] = " << rf[j] << endl;    
        }
        for(size_t j=0;j<nc;j++){
            rc[j] = y[nf+j]-acb.f(nf+j);
            //cout << "rc[j] = " << rc[j] << endl;    
        }
        // Update varaince trees
#ifdef _OPENMPI
        if(mpirank==0) psbmf.draw(gen); else psbmf.draw_mpislave(gen);
        if(mpirank==0) psbmc.draw(gen); else psbmc.draw_mpislave(gen);
#else
        psbmf.draw(gen);
        psbmc.draw(gen);
#endif
        // Update the value of sigma in sig -- replaces the wrapper
        for(size_t j=0;j<nf;j++){
            sig[j] = psbmf.f(j);
            //cout << "sigf[j] = " << sig[j] << endl;   
        }
        for(size_t j=0;j<nc;j++){
            sig[nf+j] = psbmc.f(j); 
            //cout << "sigc[j] = " << sig[nf+j] << endl;   
        }
        
        // Update calibration parameters

#ifdef _OPENMPI
        // Reset csumwr2 and nsumwr2
        csumwr2 = 0.0; 
        nsumwr2 = 0.0;
        cprop = 0.0; nprop = 0.0;
        // Get current weighted sum of residuals squared (only need field obs)
        for(size_t j=0;j<nf;j++){csumwr2+=(acb.r(j)/sig[j])*(acb.r(j)/sig[j]);}

        // Get joint proposal and update xf_copy with new u
        if(proptype=="default"){
            if(mpirank==0) uvec.drawnew(gen); else uvec.drawnew_mpi(gen);
        }else if(proptype=="mala"){
            // Get gradient, proposed move, and proposal prob
            xf_grad = xf; // update xf_grad
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.ucur);
            if(mpirank==0) uvec.drawnew_mala(grad,gen); else uvec.drawnew_mpi(gen); //mpi function is the exact same
            if(mpirank==0) cprop = uvec.logprp_mala(uvec.ucur, uvec.unew, grad);
        }
        if(mpirank>0) uvec.updatex(xf_copy,ucols,p,nf);

        // Get predictions for field obs with new u
        if(mpirank>0) acb.predict_vec(&di_prop,&fif);

        // Get new weight sum of residuals squared
        for(size_t j=0;j<nf;j++){nsumwr2+=((yf[j]-fprop[j])/sig[j])*((yf[j]-fprop[j])/sig[j]);}
        // MALA proposals
        if(proptype == "mala"){
            // get gradient for proposal
            xf_grad = xf_copy;
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.unew);
            if(mpirank>0) {nprop = uvec.logprp_mala(uvec.unew, uvec.ucur, grad);}

            // Update csumwr2 and nsumwr2 to include proposal effect
            csumwr2 += cprop;
            nsumwr2 += nprop;
        }
        // Now do the MH Step
        uvec.mhstep(csumwr2,nsumwr2,gen);
        
        // Update the x data if the propsed move was accepted
        if(uvec.accept){
            uvec.updatex(x,ucols,p,nf); // used in the mean model
            uvec.updatex(xf,ucols,p,nf); // used in the f variance model
        } 
        

#else
        // Reset csumwr2 and nsumwr2
        csumwr2 = 0.0; 
        nsumwr2 = 0.0;
        cprop = 0.0; nprop = 0.0;
        // Get current weighted sum of residuals squared (only need field obs)
        for(size_t j=0;j<nf;j++){csumwr2+=(acb.r(j)/sig[j])*(acb.r(j)/sig[j]);}
        // Get joint proposal and update xf_copy with new u
        if(proptype=="default"){
            uvec.drawnew(gen);
        }else if(proptype=="mala"){
            // Get gradient, proposed move, and proposal prob
            xf_grad = xf; // update xf_grad
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.ucur);
            uvec.drawnew_mala(grad,gen); 
            cprop = uvec.logprp_mala(uvec.ucur, uvec.unew, grad);
        }
        uvec.updatex(xf_copy,ucols,p,nf);
        
        // Get predictions for field obs with new u
        acb.predict_vec(&di_prop,&fif);
        // Get new weight sum of residuals squared
        for(size_t j=0;j<nf;j++){nsumwr2+=((yf[j]-fprop[j])/sig[j])*((yf[j]-fprop[j])/sig[j]);}
        // MALA proposal probabilities
        if(proptype == "mala"){
            // get gradient for proposal
            xf_grad = xf_copy;
            grad = acb.klgradient(nf,gradstep,xf_grad,ucols,uvec.unew);
            nprop = uvec.logprp_mala(uvec.unew, uvec.ucur, grad);

            // Update csumwr2 and nsumwr2 to include proposal effect
            csumwr2 += cprop;
            nsumwr2 += nprop;
        }
        // Now do the MH Step
        uvec.mhstep(csumwr2,nsumwr2,gen);

        // Update the x data if the propsed move was accepted
        if(uvec.accept){
            uvec.updatex(x,ucols,p,nf); // used in the mean model
            uvec.updatex(xf,ucols,p,nf); // used in the f variance model
        }
#endif
        if(mpirank==0) {
            acb.savetree_vec(i,m,onn,oid,ovar,oc,otheta); 
            psbmf.savetree(i,mh,sfnn,sfid,sfvar,sfc,sftheta);
            psbmc.savetree(i,mh,scnn,scid,scvar,scc,sctheta);
            udraws.insert(udraws.end(),uvec.ucur.begin(),uvec.ucur.end());
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
    if(mpirank==0){
        cout << "Returning posterior, please wait...";
        std::vector<int>* e_ots=new std::vector<int>(nd*m);
        std::vector<int>* e_oid=new std::vector<int>;
        std::vector<int>* e_ovar=new std::vector<int>;
        std::vector<int>* e_oc=new std::vector<int>;
        std::vector<double>* e_otheta=new std::vector<double>;
        std::vector<int>* e_sfts=new std::vector<int>(nd*mh);
        std::vector<int>* e_sfid=new std::vector<int>;
        std::vector<int>* e_sfvar=new std::vector<int>;
        std::vector<int>* e_sfc=new std::vector<int>;
        std::vector<double>* e_sftheta=new std::vector<double>;
        std::vector<int>* e_scts=new std::vector<int>(nd*mh);
        std::vector<int>* e_scid=new std::vector<int>;
        std::vector<int>* e_scvar=new std::vector<int>;
        std::vector<int>* e_scc=new std::vector<int>;
        std::vector<double>* e_sctheta=new std::vector<double>;
        for(size_t i=0;i<nd;i++){
            for(size_t j=0;j<m;j++) {
                e_ots->at(i*m+j)=static_cast<int>(oid[i*m+j].size());
                e_oid->insert(e_oid->end(),oid[i*m+j].begin(),oid[i*m+j].end());
                e_ovar->insert(e_ovar->end(),ovar[i*m+j].begin(),ovar[i*m+j].end());
                e_oc->insert(e_oc->end(),oc[i*m+j].begin(),oc[i*m+j].end());
                e_otheta->insert(e_otheta->end(),otheta[i*m+j].begin(),otheta[i*m+j].end());
            }
        }
        for(size_t i=0;i<nd;i++){
            for(size_t j=0;j<mh;j++) {
                e_sfts->at(i*mh+j)=static_cast<int>(sfid[i*mh+j].size());
                e_sfid->insert(e_sfid->end(),sfid[i*mh+j].begin(),sfid[i*mh+j].end());
                e_sfvar->insert(e_sfvar->end(),sfvar[i*mh+j].begin(),sfvar[i*mh+j].end());
                e_sfc->insert(e_sfc->end(),sfc[i*mh+j].begin(),sfc[i*mh+j].end());
                e_sftheta->insert(e_sftheta->end(),sftheta[i*mh+j].begin(),sftheta[i*mh+j].end());
            }

            for(size_t j=0;j<mh;j++) {
                e_scts->at(i*mh+j)=static_cast<int>(scid[i*mh+j].size());
                e_scid->insert(e_scid->end(),scid[i*mh+j].begin(),scid[i*mh+j].end());
                e_scvar->insert(e_scvar->end(),scvar[i*mh+j].begin(),scvar[i*mh+j].end());
                e_scc->insert(e_scc->end(),scc[i*mh+j].begin(),scc[i*mh+j].end());
                e_sctheta->insert(e_sctheta->end(),sctheta[i*mh+j].begin(),sctheta[i*mh+j].end());
            }
        }
        //write out to file
        std::ofstream omf(folder + modelname + ".fit");
        omf << nd << endl;
        omf << m << endl;
        omf << mh << endl;
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
        
        omf << e_sfts->size() << endl;
        for(size_t i=0;i<e_sfts->size();i++) omf << e_sfts->at(i) << endl;
        omf << e_sfid->size() << endl;
        for(size_t i=0;i<e_sfid->size();i++) omf << e_sfid->at(i) << endl;
        omf << e_sfvar->size() << endl;
        for(size_t i=0;i<e_sfvar->size();i++) omf << e_sfvar->at(i) << endl;
        omf << e_sfc->size() << endl;
        for(size_t i=0;i<e_sfc->size();i++) omf << e_sfc->at(i) << endl;
        omf << e_sftheta->size() << endl;
        for(size_t i=0;i<e_sftheta->size();i++) omf << std::scientific << e_sftheta->at(i) << endl;
        
        omf << e_scts->size() << endl;
        for(size_t i=0;i<e_scts->size();i++) omf << e_scts->at(i) << endl;
        omf << e_scid->size() << endl;
        for(size_t i=0;i<e_scid->size();i++) omf << e_scid->at(i) << endl;
        omf << e_scvar->size() << endl;
        for(size_t i=0;i<e_scvar->size();i++) omf << e_scvar->at(i) << endl;
        omf << e_scc->size() << endl;
        for(size_t i=0;i<e_scc->size();i++) omf << e_scc->at(i) << endl;
        omf << e_sctheta->size() << endl;
        for(size_t i=0;i<e_sctheta->size();i++) omf << std::scientific << e_sctheta->at(i) << endl;

        omf.close();

        //Write calibration parameter -- files
        std::ofstream ouf(folder + modelname + ".udraws");
        for(size_t i=0;i<udraws.size();i++) ouf << udraws.at(i) << endl;
        ouf.close();
        cout << " done." << endl;
    
    }
#ifdef _OPENMPI
   delete[] lwr;
   delete[] upr;
   MPI_Finalize();
#endif
    return 0;
}
