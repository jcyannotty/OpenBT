#include <iostream>
#include <fstream>

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mxbrt.h"
#include "amxbrt.h"

//Include Eigen library
#include "Eigen/Dense"

#ifdef _OPENMPI
#   include <mpi.h>
#endif

int main(){
    //-------------------------------------------------------
    //---Read in Data for mxbrt examples
    //-------------------------------------------------------
    crn gen;
    gen.set_seed(199);

    int tc=4; //thread count for OpenMP

    //--------------------------------------------------
    //read in y
    std::vector<double> y;
    double ytemp;

    std::ifstream yf("y.txt");
    while(yf >> ytemp)
        y.push_back(ytemp);
    size_t n = y.size();
    cout << "n from y.txt: " << n <<endl;

    //--------------------------------------------------
    //read in x
    std::vector<double> x;
    double xtemp;
    size_t p;
    p=1;

    std::ifstream xf("x.txt");
    while(xf >> xtemp){
        x.push_back(xtemp);
    }
    
    if(x.size() != n*p) {
        cout << "error: input x file has wrong number of values\n";
        return 1;
    }
    cout << "n,p: " << n << ", " << p << endl;

    //--------------------------------------------------
    //read in f
    std::vector<double> f;
    double ftemp;
    size_t k; //number of columns in f
    k=2;

    std::ifstream ff("f.txt");
    while(ff >> ftemp){
        f.push_back(ftemp);
    }
    
    if(f.size() != n*k) {
        cout << "error: input f file has wrong number of values\n";
        return 1;
    }
    cout << "n,k: " << n << ", " << k << endl;
    
    //--------------------------------------------------
    //Make dinfo and diterator
    dinfo di;
    di.n=n;di.p=p,di.x = &x[0];di.tc=tc;
    di.y = &y[0];

    diterator diter(&di);

    //--------------------------------------------------
    //make xinfo
    xinfo xi;
    size_t nc=100; //100
    makexinfo(p,n,&x[0],xi,nc); //use the 1st column with x[0]

    prxi(xi);
    //--------------------------------------------------
    //make finfo -- need to read in and store f formally, just using same x from above for now
    finfo fi;
    makefinfo(k, n, &f[0], fi);
    cout << fi << endl;

    //--------------------------------------------------
    // read in the initial change of variable rank correlation matrix
    std::vector<std::vector<double> > chgv;
    std::vector<double> cvvtemp;
    double cvtemp;
    std::ifstream chgvf("chgv.txt");
    for(size_t i=0;i<di.p;i++) {
        cvvtemp.clear();
        for(size_t j=0;j<di.p;j++) {
            chgvf >> cvtemp;
            cvvtemp.push_back(cvtemp);
        }
        chgv.push_back(cvvtemp);
    }
    cout << "change of variable rank correlation matrix loaded:" << endl;
    for(size_t i=0;i<di.p;i++) {
        for(size_t j=0;j<di.p;j++)
            cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
        cout << endl;
    }

    //--------------------------------------------------
    //Make test set information
    //Read in test data
    //read in x
    std::vector<double> x_test;
    int n_test;
    std::ifstream xf2("xtest.txt");
    while(xf2 >> xtemp){
        x_test.push_back(xtemp);
    }
    n_test = x_test.size()/p;
    if(x_test.size() != n_test*p) {
        cout << "error: input x file has wrong number of values\n";
        return 1;
    }
    cout << "n_test,p: " << n_test << ", " << p << endl;

    //read in f
    std::vector<double> f_test;
    std::ifstream ff2("ftest.txt");
    while(ff2 >> ftemp){
        f_test.push_back(ftemp);
    }
    
    if(f_test.size() != n_test*k) {
        cout << "error: input f file has wrong number of values\n";
        return 1;
    }
    cout << "n_test,k: " << n_test << ", " << k << endl;
    
    //Make dinfo and diterator
    dinfo di_test;
    std::vector<double> y_test(n_test); //empty y
    for(int j=0;j<n_test;j++){y_test.push_back(0);}
    di_test.n=n_test;di_test.p=p,di_test.x = &x_test[0];di_test.tc=tc;di_test.y = &y_test[0];    
    
    diterator diter_test(&di_test);

    //Make finfo
    finfo fi_test;
    makefinfo(k,n_test, &f_test[0], fi_test);

    //-------------------------------------------------------
    //Example 1 -- Test Constructors
    //-------------------------------------------------------
    amxbrt am1; //default constructor
    amxbrt am2(3); //Constructor with number of trees as input

    //Print am2 model 
    am2.pr_vec();

    //-------------------------------------------------------
    //Example 2 -- Test MCMC
    //-------------------------------------------------------
    /*
    cout << "\n\n-----------------------------------------" << endl;
    cout << "Example 2: Work with a mxbrt object \n" << endl;
    
    //Initialize prior parameters
    int m = 5; //25
    double *sig = new double[di.n];
    double tau = 0.5/sqrt(double(m)); // --- 0.2 with 2 trees
    double beta0 = 0.55/double(m); // --- 0.2 with 2 trees
    for(size_t i=0;i<di.n;i++) sig[i]=0.03;

    //First mix bart object with basic constructor
    amxbrt axb(m); //m Trees 
    cout << "****Initial Object" << endl; 
    axb.pr_vec();
    axb.setxi(&xi);    //set the cutpoints for this model object 
    //function output 
    axb.setfi(&fi,k); //set function output for this model object
    //data objects
    axb.setdata_mix(&di);  //set the data for model mixing
    //thread count
    axb.settc(tc);      //set the number of threads when using OpenMP, etc.
    //tree prior
    axb.settp(0.95, //the alpha parameter in the tree depth penalty prior
            0.75     //the beta parameter in the tree depth penalty prior
            );
    //MCMC info
    axb.setmi(
            0.5,  //probability of birth/death
            0.5,  //probability of birth
            3,    //minimum number of observations in a bottom node
            true, //do perturb/change variable proposal?
            0.01,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            0.01,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv  //initialize the change of variable correlation matrix.
            );
    axb.setci(tau,beta0,sig);

    cout << "\n*****After init:\n";
    axb.pr_vec();

    cout << "\n-----------------------------------" << endl;
    cout << "-----------------------------------" << endl;

    size_t nd = 20000;
    size_t nadapt=5000;
    size_t adaptevery=500;
    size_t nburn=1000;
    std::vector<double> fitted(n), predicted(n_test);
    dinfo di_predict;
    di_predict.n=n_test;di_predict.p=p,di_predict.x = &x_test[0];di_predict.tc=tc;di_predict.y = &predicted[0];
    for(size_t i=0;i<nadapt;i++) { axb.drawvec(gen); if((i+1)%adaptevery==0) axb.adapt(); }
    for(size_t i=0;i<nburn;i++) axb.drawvec(gen); 
    
    cout << "\n*****After "<< nd << " draws:\n";
    cout << "Collecting statistics" << endl;
    axb.setstats(true);
    for(int i = 0; i<nd; i++){
        if((i % 500) ==0){cout << "***Draw " << i << "\n" << endl;}
        axb.drawvec(gen);
        for(size_t j=0;j<n;j++) fitted[j]+=axb.f(j)/nd;
        //if((i % 5) ==0){cout << "Draw " << i+1 << " --- " << "F(10) = " << axb.f(10) << " --- R(10) = " << axb.r(10) << endl;}
        axb.predict_mix(&di_test, &fi_test);
        di_predict += di_test;
    }
    
    //Take the prediction average
    di_predict/=((double)nd);

    // summary statistics
    unsigned int varcount[p];
    unsigned int totvarcount=0;
    for(size_t i=0;i<p;i++) varcount[i]=0;
    unsigned int tmaxd=0;
    unsigned int tmind=0;
    double tavgd=0.0;

    axb.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
    for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
    tavgd/=(double)(nd*m);

    cout << "Average tree depth: " << tavgd << endl;
    cout << "Maximum tree depth: " << tmaxd << endl;
    cout << "Minimum tree depth: " << tmind << endl;
    cout << "Variable perctg:    ";
    for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
    cout << endl;
    cout << "                    ";
    for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
    cout << endl;

    
    cout << "Print Fitted Values" << endl;
    for(int i = 0; i<n; i++){
        cout << "X = " << x[i] << " -- Y = " << y[i] <<" -- Fitted " << fitted[i] << " -- Error = " << fitted[i] - y[i] << endl;
    }

    cout << "Print Predicted Values" << endl;
    for(int i = 0; i<n_test; i++){
        cout << i <<" -- Predicted " << predicted[i] << endl;
    }    
    
    //Write all data values to a file
    std::ofstream outdata;
    outdata.open("fit_amxb2_m20.txt"); // opens the file
    if( !outdata ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    for(int i = 0; i<n; i++){
        outdata << fitted[i] << endl;
    }
    outdata.close();

    //Write all data values to a file
    std::ofstream outpred;
    outpred.open("predict_amxb2_m20.txt"); // opens the file
    if( !outpred ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    for(int i = 0; i<n_test; i++){
        outpred << predicted[i] << endl;
    }
    outpred.close();
    
    */
   
    //-------------------------------------------------------
    //Example 3 -- Test MCMC with unknown constant variance
    //-------------------------------------------------------
    cout << "\n\n-----------------------------------------" << endl;
    cout << "Example 3: Work with a mxbrt object \n" << endl;
    
    //Initialize prior parameters
    int m = 5;
    double *sig = new double[di.n];
    double tau = 0.5/sqrt(double(m));
    double beta0 = 0.55/double(m);
    double nu = 5.0;
    double lambda = 0.01;
    for(size_t i=0;i<di.n;i++) sig[i]=0.1;
        
    //First mix bart object with basic constructor
    amxbrt axb(m); //20 Trees 
    cout << "****Initial Object" << endl; 
    axb.pr_vec();
    axb.setxi(&xi);    //set the cutpoints for this model object 
    //function output 
    axb.setfi(&fi,k); //set function output for this model object
    //data objects
    axb.setdata_mix(&di);  //set the data for model mixing
    //thread count
    axb.settc(tc);      //set the number of threads when using OpenMP, etc.
    //tree prior
    axb.settp(0.95,0.75);//the alpha and beta parameters in the tree depth penalty prior
    //MCMC info
    axb.setmi(
            0.5,  //probability of birth/death
            0.5,  //probability of birth
            1,    //minimum number of observations in a bottom node
            true, //do perturb/change variable proposal?
            0.01,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            0.01,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv  //initialize the change of variable correlation matrix.
            );
    axb.setci(tau,beta0,sig);
    axb.setvi(nu, lambda);

    cout << "\n*****After init:\n";
    axb.pr_vec();

    cout << "\n-----------------------------------" << endl;
    cout << "-----------------------------------" << endl;

    size_t nd = 20000;
    size_t nadapt=5000;
    size_t adaptevery=500;
    size_t nburn=1000;
    std::vector<double> fitted(n), predicted(n_test);
    dinfo di_predict;
    std::ofstream outdraw; //used for final 500 prediction draws
    
    di_predict.n=n_test;di_predict.p=p,di_predict.x = &x_test[0];di_predict.tc=tc;di_predict.y = &predicted[0];
    for(size_t i=0;i<nadapt;i++) { axb.drawvec(gen); if((i+1)%adaptevery==0) axb.adapt(); }
    for(size_t i=0;i<nburn;i++) axb.drawvec(gen); 
    
    //Initialize the sigma posterior txt file
    std::ofstream outsig;
    outsig.open("postsig_axb2.txt"); // opens the file
    outsig.close(); // closes the file

    cout << "\n*****After "<< nd << " draws:\n";
    cout << "Collecting statistics" << endl;
    axb.setstats(true);
    for(int i = 0; i<nd; i++){
        if((i % 1000) ==0){cout << "***Draw " << i << "\n" << endl;}
        
        //Draw tree and theta -- then get fitted values
        axb.drawvec(gen);
        for(size_t j=0;j<n;j++) fitted[j]+=axb.f(j)/nd;
        
        //Draw Sigma and save the last 25% of draws
        axb.drawsigma(gen);
        if(i > nd*0.75){
            outsig.open("postsig_axb2.txt", std::ios_base::app); // opens the file
            outsig << axb.getsigma() << endl;
            outsig.close(); // closes the file
        }
        //Get Predictions
        axb.predict_mix(&di_test, &fi_test);
        di_predict += di_test;

        //Write last 500 posterior draws to txt file
        /*
        if(i == nd-500){
            outdraw.open("pdraws_amxb2_sig.txt"); // opens the file
            for(int i = 0; i<n_test; i++){
                outdraw << y_test[i] << ","; //write the current mixed function to a text file
            }
            outdraw << endl;
            outdraw.close();
        }else if(i > nd-500){
            outdraw.open("pdraws_amxb2_sig.txt",std::ios_base::app); // opens and appends the file
            for(int i = 0; i<n_test; i++){
                outdraw << y_test[i] << ","; //write the current mixed function to a text file
            }
            outdraw << endl;
            outdraw.close();
        }
        */
    }
    
    //Take the prediction average
    di_predict/=((double)nd);

    // summary statistics
    unsigned int varcount[p];
    unsigned int totvarcount=0;
    for(size_t i=0;i<p;i++) varcount[i]=0;
    unsigned int tmaxd=0;
    unsigned int tmind=0;
    double tavgd=0.0;

    axb.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
    for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
    tavgd/=(double)(nd);

    cout << "Average tree depth: " << tavgd << endl;
    cout << "Maximum tree depth: " << tmaxd << endl;
    cout << "Minimum tree depth: " << tmind << endl;
    cout << "Variable perctg:    ";
    for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
    cout << endl;
    cout << "                    ";
    for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
    cout << endl;

    cout << "Print Fitted Values" << endl;
    for(int i = 0; i<n; i++){
        cout << "X = " << x[i] << " -- Y = " << y[i] <<" -- Fitted " << fitted[i] << " -- Error = " << fitted[i] - y[i] << endl;
    }

    cout << "Print Predicted Values" << endl;
    for(int i = 0; i<n_test; i++){
        cout << i <<" -- Predicted " << predicted[i] << endl;
    }

    //Print the Last Tree
    axb.pr_vec();

    //Write all data values to a file
    std::ofstream outdata;
    outdata.open("fit_amxb2.txt"); // opens the file
    if( !outdata ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    for(int i = 0; i<n; i++){
        outdata << fitted[i] << endl;
    }
    outdata.close();

   //Write all data values to a file
    std::ofstream outpred;
    outpred.open("predict_amxb2.txt"); // opens the file
    if( !outpred ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    for(int i = 0; i<n_test; i++){
        outpred << predicted[i] << endl;
    }
    outpred.close();


    //-------------------------------------------------------
    //Example 4 -- Do some draws and look at the results
    //-------------------------------------------------------
    /*
    cout << "\n\n-----------------------------------------" << endl;
    cout << "Example 4: View some results \n" << endl;
    
    //Initialize prior parameters
    int m = 10; //25
    double *sig = new double[di.n];
    double tau =  1.0; //0.5/sqrt(double(m));
    double beta0 = 0.0; //0.53/double(m);
    for(size_t i=0;i<di.n;i++) sig[i]=0.03;

    //First mix bart object with basic constructor
    amxbrt axb(m); //20 Trees 
    cout << "****Initial Object" << endl; 
    axb.pr_vec();
    axb.setxi(&xi);    //set the cutpoints for this model object 
    //function output 
    axb.setfi(&fi,k); //set function output for this model object
    //data objects
    axb.setdata_mix(&di);  //set the data for model mixing
    //thread count
    axb.settc(tc);      //set the number of threads when using OpenMP, etc.
    //tree prior
    axb.settp(0.95, //the alpha parameter in the tree depth penalty prior
            0.75     //the beta parameter in the tree depth penalty prior
            );
    //MCMC info
    axb.setmi(
            0.5,  //probability of birth/death
            0.5,  //probability of birth
            3,    //minimum number of observations in a bottom node
            true, //do perturb/change variable proposal?
            0.01,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            0.01,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv  //initialize the change of variable correlation matrix.
            );
    axb.setci(tau,beta0,sig);

    cout << "\n*****After init:\n";
    axb.pr_vec();

    cout << "\n-----------------------------------" << endl;
    cout << "-----------------------------------" << endl;

    size_t nd = 10;
    size_t nadapt=5000;
    size_t adaptevery=500;
    size_t nburn=1000;
    //for(size_t i=0;i<nadapt;i++) { axb.drawvec(gen); if((i+1)%adaptevery==0) axb.adapt(); }
    //for(size_t i=0;i<nburn;i++) axb.drawvec(gen); 
    
    cout << "\n*****After "<< nd << " draws:\n";
    cout << "Collecting statistics" << endl;
    axb.setstats(true);
    for(int i = 0; i<nd; i++){
        cout << "\n------------------------------" << endl;
        cout << "~~Draw " << i+1 << endl;
        axb.drawvec(gen);
        axb.pr_vec();
    }
    */
    return 0;

}